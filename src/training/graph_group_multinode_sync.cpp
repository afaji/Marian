#include <cuda.h>
#include "training/graph_group_multinode_sync.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"


#include "training/gradient_dropping/dropper.h"
#include "training/gradient_dropping/sparse_tensor.h"

namespace marian {


void MultiNodeGraphGroupSync::performUpdate(){
  // Perform optimizer step
  syncOptimizer_->update(clientGraphs_.back());

  if(movingAvg_)
  	updateAvgParams(paramsAvg_,
                    clientGraphs_.back()->params()->vals(),
                    scheduler_->numberOfBatches());
}

void MultiNodeGraphGroupSync::nodeParamSync(){
  //Distribute the graph to the rest of the devices
  std::vector<std::thread> threads;
  for(int idx = 0; idx < devices_.size() - 1; idx++) {
    threads.emplace_back(std::thread(
        [=](int idx) {
          clientGraphs_[idx]->params()->vals()->copyFrom(
            clientGraphs_.back()->params()->vals());
        },
        idx));
  }
  for(auto&& t : threads) {
    t.join();
  }
}




void MultiNodeGraphGroupSync::updateAvgParams(Tensor paramsAvg,
                                              Tensor params,
                                              size_t batches) {
  using namespace functional;
  float decay
      = std::max(mvDecay_, 1.f - (float)(batches + 1) / (float)(batches + 10));
  Element(_1 = ((1.f - decay) * _1) + (decay * _2), paramsAvg, params);
}

/**
 * Set given scheduler to register training observers on the shard optimizers.
 */
void MultiNodeGraphGroupSync::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see a change of learning rate
  scheduler_->registerTrainingObserver(scheduler_);

  scheduler_->registerTrainingObserver(syncOptimizer_);
}

/**
 * Allocate new tensor on given GPU and store allocator.
 */
Tensor MultiNodeGraphGroupSync::newTensor(int size, Ptr<Backend> backend) {
  Tensor t;
  Ptr<TensorAllocator> allocator = New<TensorAllocator>(backend);
  allocator->reserveExact(size * sizeof(float));
  allocator->allocate(t, {1, size});
  allocators_.push_back(allocator);
  return t;
}

/**
 * Setup training environment and launch server thread and (if enabled) client
 * communication overlap threads.
 * Includes setting up MPI, node and shard sizes, clients, server shards and
 * communication overlap stuff.
 */
void MultiNodeGraphGroupSync::init(Ptr<data::Batch> batch) {

  // Setup clients and shards
  setupClients(batch);
  int network_size = (int)clientGraphs_[0]->params()->vals()->size();

  LOG(info, "model size = {} float params", network_size);
  if(movingAvg_) {
    paramsAvg_ = newTensor(network_size, clientGraphs_.back()->getBackend());  
    paramsAvg_->copyFrom(clientGraphs_[0]->params()->vals());
  
    LOG(info, "init paramsAvg {}", paramsAvg_->get(77));
  }

  // setup sync sgd storage, We keep the summed gradient on Node 0
  sumGradientBuffer = newTensor(network_size, clientGraphs_[0]->getBackend());
  accGradientsSync = newTensor(network_size, clientGraphs_[0]->getBackend());
  accGradientsSync->set(0);
  sumGradientBuffer->set(0);

  // init for dropping
  if (droping_rate > 0.0) {
    int sparse_size = std::max(network_size * 0.1 ,
                               network_size * (1.0 - droping_rate));
    sparseGradient = SparseTensor(
          new SparseTensorBase(sparse_size * 1.2,
                               accGradientsSync->getBackend()));
    sparseGather = SparseTensor(
          new SparseTensorBase(sparse_size * mpi_->numMPIProcesses(),
                               clientGraphs_.back()->getBackend()));

    dropper = PrepareGradientDrop(accGradientsSync->getBackend()->getDeviceId());
  }
}

/**
 * Initialize the CPU arrays, with pinned memory for faster CudaMemCpy
 * operations. Requires the graph to be initialized first so we know its size
 */
void MultiNodeGraphGroupSync::initCPUArrays() {
  int network_size = clientGraphs_[0]->params()->vals()->size();

  accGradientsSync_cpu
      = std::vector<float>(clientGraphs_[0]->params()->vals()->size());
  receiveBuffer_cpu
      = std::vector<float>(clientGraphs_[0]->params()->vals()->size());
  // inits for gradient dropping
  if (droping_rate > 0.0) {
    int sparse_size = std::max(network_size * 0.1 ,
                               network_size * (1.0 - droping_rate));
    sparseGrad_cpu = std::vector<float>(sparse_size);
    sparseIndices_cpu = std::vector<int>(sparse_size);


    gatherGrads_cpu = std::vector<float>(sparse_size * 
                                        mpi_->numMPIProcesses());

    gatherIndices_cpu = std::vector<int>(sparse_size * 
                                        mpi_->numMPIProcesses());
  }
}

/**
 * Setup clients that will compute gradients and communicate them with the
 * server shards.
 * There is one client per GPU.
 */
void MultiNodeGraphGroupSync::setupClients(Ptr<data::Batch> batch) {
  runBatchThroughClientGraphs(batch);
  initCPUArrays();
}

/**
 * Initialize the graphs (models) of all clients on this node with the given
 * batch.
 */
void MultiNodeGraphGroupSync::runBatchThroughClientGraphs(
    Ptr<data::Batch> batch) {
  std::string name = options_->get<std::string>("model");

    for(int i = 0; i < devices_.size(); i++) {
      THREAD_GUARD(clientBuilders_[i]->build(clientGraphs_[i], batch);
                 clientGraphs_[i]->forward();
                 
      clientGraphs_[i]->params()->allocateBackward();
      clientGraphs_[i]->params()->set_zero_adjoint();
      );
    }
   if(filesystem::exists(name)) {
    std::string nameGraph = name;
    if(filesystem::exists(name + ".orig.npz"))
        // Load the original parameters from model.npz.orig.npz
      nameGraph += ".orig.npz";
     
    size_t i = 0;
    for(auto graph : clientGraphs_)
      clientBuilders_[i++]->load(graph, nameGraph); // we just load it N times from disk (it'll be in disk cache after the first)
  } 
}

/**
 * Initialize variables required for overlapping client computations and
 * communication.
 * Includes summed and committed word counts, buffer flags, mutexes and
 * condition variables.
 */
void MultiNodeGraphGroupSync::sumGRAD(Tensor gradient) {
  std::lock_guard<std::mutex> guard(sumGradientMutex_);
  sumGradientBuffer->copyFrom(gradient);
  using namespace functional;  //@TODO makes more sense to do that on the CPU i
                               // think
  Element(_1 += _2, accGradientsSync, sumGradientBuffer);
}


/**
 * If it's rank 0, it's a local update, if it's rank one it's remote
 * send and receive. Make sure you only call from device 0.
 */
void MultiNodeGraphGroupSync::sendReceiveUpdateSparse() {
  #if MPI_FOUND

  int network_size = accGradientsSync->size();
  int sparse_limit = sparseGrad_cpu.size();
  float curr_droping_rate = droping_rate;

  static int step = 0;
  int WARMUP = 1000;
  int WARMUP2 = 1000;

  if (true && scheduler_->numberOfBatches() < WARMUP) {
    curr_droping_rate = std::pow(droping_rate, WARMUP / (1.0 + scheduler_->numberOfBatches()));
    // LOG(info, "WARMUP DROPPING = {}", curr_droping_rate);
  }

  int sparse_size = network_size * (1.0 - curr_droping_rate) * 1.2;
  static float resize_steps[4] = {0.95, 0.98, 0.99, 1.0};
  static int step_cnt = 0;
  if (sparse_size < sparseGrad_cpu.size() && curr_droping_rate >= resize_steps[step_cnt]) {
    step_cnt++;
    LOG(info, "resizing sparse gradient size to {}", sparse_size);
    sparseGrad_cpu.resize(sparse_size);
    sparseIndices_cpu.resize(sparse_size * mpi_->numMPIProcesses());
  }
 
  // START OF NO COMMUNICATION
  bool local = true;
  std::string local_mode = "replace";
  
  if ((scheduler_->numberOfBatches() < WARMUP2) || sparse_size > sparse_limit){
     sendReceiveUpdateSync(accGradientsSync);
     return; 
  }
 
  dropper->dropGraph(accGradientsSync,
                     clientGraphs_[0]->params()->grads(),
                     sparseGradient,
                     curr_droping_rate,
                     dropping_momentum);
  
  if (scheduler_->numberOfBatches() < WARMUP2 || sparse_size > sparse_limit){
    sendReceiveUpdateSync(accGradientsSync);
    return;
  }

  // Copy the gradient and indices to CPU
  sparseGradient->get(sparseGrad_cpu, sparseIndices_cpu, sparse_size);

  static MPI_Request r1, r2;
  // Gather gradient
  MPI_Iallgather(sparseGrad_cpu.data(), sparse_size, MPI_FLOAT,
    gatherGrads_cpu.data(), sparse_size, MPI_FLOAT,
    MPI_Comm MPI_COMM_WORLD, &r1);

  // Gather indices
  MPI_Iallgather(sparseIndices_cpu.data(), sparse_size, MPI_INT,
    gatherIndices_cpu.data(), sparse_size, MPI_INT,
    MPI_Comm MPI_COMM_WORLD, &r2);

  //parallel while data transfer is happening:
  if (local) {
    using namespace functional; //@TODO makes more sense to do that on the CPU i think
    //replace
    if (local_mode == "replace") {
      Element(_1 -= _2, accGradientsSync, clientGraphs_[0]->params()->grads());
    }
    //sum or replace
    clientGraphs_.back()->params()->grads()->copyFrom(accGradientsSync);
    // scaling
    Element(_1 *= 2, clientGraphs_.back()->params()->grads()); 
    // no error feed
    // dropper->error()->set(0);
  }
  
  MPI_Wait(&r1, MPI_STATUS_IGNORE);
  MPI_Wait(&r2, MPI_STATUS_IGNORE);
  

  // Update params
  // Copy the data back to the GPU and do optimizer update
  sparseGather->set(gatherGrads_cpu, gatherIndices_cpu, sparse_size * mpi_->numMPIProcesses());
  // if use local context
  if (local) {
    using namespace functional;
    sparseGather->scatterAdd(clientGraphs_.back()->params()->grads(), 0);
  }
  else
  // if default 
    sparseGather->toDense(clientGraphs_.back()->params()->grads(), 0);


  // END OF COMMUNICATION
  // clientGraphs_.back()->params()->grads()->copyFrom(accGradientsSync);
  
  performUpdate();

  if (local && step++ % 500 == 0) {
    clientGraphs_.back()->params()->vals()->get(accGradientsSync_cpu);
    MPI_Allreduce(accGradientsSync_cpu.data(), //CPU buffers
              receiveBuffer_cpu.data(),
              network_size,
              MPI_FLOAT,
              MPI_SUM,
              MPI_COMM_WORLD);
    clientGraphs_.back()->params()->vals()->set(receiveBuffer_cpu);
    using namespace functional; //@TODO makes more sense to do that on the CPU i think
    Element(_1 /= mpi_->numMPIProcesses(), clientGraphs_.back()->params()->vals());
  }

  nodeParamSync();
  
  accGradientsSync->set(0);
  std::fill(sparseGrad_cpu.begin(), sparseGrad_cpu.end(), 0);
  std::fill(sparseIndices_cpu.begin(), sparseIndices_cpu.end(), 0);

  #endif
}


/**
 * If it's rank 0, it's a local update, if it's rank one it's remote
 * send and receive. Make sure you only call from device 0.
 */
void MultiNodeGraphGroupSync::sendReceiveUpdateSync(Tensor grad) {
#if MPI_FOUND
  auto network_size = accGradientsSync_cpu.size(); // @TODO: get this from accGradientSync (not CPU), it is more direct

  // Copy the data to the CPU
  grad->get(accGradientsSync_cpu);

  // Wait until all nodes are ready
  
  MPI_Barrier(MPI_COMM_WORLD);

  bool SHARD = true;
  if (SHARD) {
    static MPI_Request r[12];
    int shard_size = network_size / mpi_->numMPIProcesses();
    int offset = 0;
    for (int i=0;i<mpi_->numMPIProcesses();i++) {
      MPI_Iallreduce(accGradientsSync_cpu.data() + offset,
                     receiveBuffer_cpu.data() + offset,
                     shard_size,
                     MPI_FLOAT, MPI_SUM, MPI_Comm MPI_COMM_WORLD, &r[i]);                     
      offset += shard_size;
    }
    for (int i=0;i<mpi_->numMPIProcesses();i++) {
       MPI_Wait(&r[i], MPI_STATUS_IGNORE);     
    }   
  } else {

    MPI_Allreduce(accGradientsSync_cpu.data(),  // CPU buffers
                  receiveBuffer_cpu.data(),
                  network_size,
                  MPI_FLOAT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
  }

  // LOG(info, "Total {}", receiveBuffer_cpu[77]);
  // Copy the data back to the GPU and do optimizer update
  // Do update with last GPU to distribute the memory
  clientGraphs_.back()->params()->grads()->set(receiveBuffer_cpu);

  performUpdate();
  nodeParamSync();

  // set the accumulating buffers to zero;
  accGradientsSync->set(0);
  std::fill(accGradientsSync_cpu.begin(), accGradientsSync_cpu.end(), 0.0f);
  std::fill(receiveBuffer_cpu.begin(), receiveBuffer_cpu.end(), 0.0f);
  // @TODO:
  //  - these fill() calls are not necessary
  //  - can accGradientsSync_cpu and receiveBuffer_cpu be the same buffer? Then change allReduce() to single-argument function.
#endif
}

/**
 * Execute given batch on this node, pushing/pulling the resulting
 * gradients/parameters to/from the server shards
 * or -- if comm. overlap enabled -- to/from the communication buffers, summing
 * gradients locally if the communication thread is busy
 *
 * @param batch Batch on which to perform forward and backward passes.
 */
void MultiNodeGraphGroupSync::execute(Ptr<data::Batch> fullBatch) {


  std::vector<Ptr<data::Batch>> batches = fullBatch->split(devices_.size());
  if(!initialized_) {
    init(batches.front());
    initialized_ = true;
  }

  static int t = -1;
  static StaticLoss loss;
  static size_t num_seen_words = 0;
  static size_t num_seen_sentences = 0;
  // accGradientsSync->set(0);

  {
    auto task = [this, batches](int my_id) {
      auto batch = batches[my_id];
      auto graph = clientGraphs_[my_id];
      auto builder = clientBuilders_[my_id];

      auto lossNode = builder->build(graph, batch);

      if(t == 0) {
        if(my_id != 0)
          graph->params()->vals()->copyFrom(clientGraphs_[0]->params()->vals());
      }

      graph->forward();
      {
        std::lock_guard<std::mutex> guard(sumCostMutex_);
        loss += *lossNode;
        num_seen_words += batch->words();
        num_seen_sentences += batch->size();
      }
      graph->backward();

      graph->getBackend()
          ->synchronize();  //@Alham do you know why we need this here?

      sumGRAD(graph->params()->grads());
    };

    ThreadPool pool(devices_.size(), devices_.size());
    for(int idx = 0; idx < devices_.size(); ++idx)
      pool.enqueue(task, idx);
  }

  t++;

  if(t % tau_ == 0){
    if (droping_rate > 0.0)
      sendReceiveUpdateSparse();
    else
      sendReceiveUpdateSync(accGradientsSync);
  }
  
  // Run scheduler (if enabled)
  if(t % tau_ == 0 && scheduler_) {
    if(tau_ > 1) {
      scheduler_->update(loss, /*numReadBatches=*/1, num_seen_sentences , num_seen_words);
    } else {
      scheduler_->update(loss, fullBatch);
    }

    num_seen_words = 0;
    num_seen_sentences = 0;
    loss.reset();
    #if MPI_FOUND
    if((scheduler_->saving() || scheduler_->validating())) {
      // wait until other nodes are ready  
      MPI_Barrier(MPI_COMM_WORLD);

      if(movingAvg_)
          accGradientsSync->copyFrom(clientGraphs_[0]->params()->vals());

      if(movingAvg_)
        for(auto graph : clientGraphs_)
          graph->params()->vals()->copyFrom(paramsAvg_);

      // TODO: Saving is broken
      if(mpi_->myMPIRank() == 0 && scheduler_->saving())
        this->save(clientGraphs_[0]);

      if(mpi_->myMPIRank() == 0 && scheduler_->validating()) {
          scheduler_->validate(clientGraphs_);
      }

      if(movingAvg_)
        for(auto graph : clientGraphs_)
          graph->params()->vals()->copyFrom(accGradientsSync);

      accGradientsSync->set(0);
      // inform other nodes to continue  
      MPI_Barrier(MPI_COMM_WORLD);

    }
    #endif
  }
}

}  // namespace marian
