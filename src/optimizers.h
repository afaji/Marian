#pragma once

#include <map>
#include <memory>
#include <boost/any.hpp>
#include "tensor_operators.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>

namespace marian {

// @TODO: modify computation graph to group all paramters in single matrix object.
// This will allow to perform a single large SGD update per batch. Currently there
// are as many updates as different parameters.


//data is a pointer to gradient, already stored in GPU.
static bool has_init;
static float *temp_d;
static float *d_error;


__global__ void grad_drop(float* data, float* errors, float* threshold_ptr, int max_size){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  float cut_off = threshold_ptr[0];
  if (idx >= max_size)
    return;
  if (abs(data[idx]) <= cut_off){
    errors[idx] = data[idx];
    data[idx] = 0;
  }else{
    errors[idx] = 0;
  }
}

__global__ void grad_add_error(float* data, float* errors, int max_size){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= max_size)
    return;
  data[idx] += errors[idx];
}



struct t_abs
{
  __host__ __device__
  void operator()(float &x)
  {
    x = abs(x);
  }
};


static int wow = 10;

void grad_drop_do(float* data, float* errors, float* tmp_data, int len, float rate){
  int threads = 512;
  int blocks = 1 + len/threads;
  grad_add_error<<<blocks, threads>>>(data, errors, len);
  thrust::device_ptr<float> dev_data_ptr(tmp_data);
  thrust::for_each(dev_data_ptr, dev_data_ptr + len, t_abs());
  thrust::sort(dev_data_ptr, dev_data_ptr + len); // OVERKILL. Too slow and need extra memory. Need to replace with faster k-th selection. (inplace Radix Select?)
  int cut_index = len * rate;
  if (cut_index >= len)
    cut_index = len -1;

  grad_drop<<<blocks, threads>>>(data, errors, tmp_data + cut_index, len);

}


void grad_drop(ExpressionGraphPtr graph, float rate){
    int f_len = graph->params().grads()->size();

    if (!has_init){
      has_init = true;
      int extra = (2 * f_len * sizeof(float)) / (1024 * 1024);
      std::cerr<<"reserving extra "<<extra<<" MB"<<std::endl;
      cudaMalloc((void **)&temp_d, f_len * sizeof(float));
      cudaMalloc((void **)&d_error, f_len * sizeof(float));
    }
    

    cudaMemcpy(temp_d,graph->params().grads()->data(), f_len * sizeof(float), cudaMemcpyDeviceToDevice);
    int pos = 0;
    for(auto& param : graph->params()){
      //std::cerr<<param->grad()->shape()[0]<<" x "<<param->grad()->shape()[1]<<std::endl;
      int len = param->grad()->size();
      grad_drop_do(param->grad()->data(), d_error + pos, temp_d + pos, len, rate);
      pos += len;
    }
}


class OptimizerBase {
  public:
    virtual void update(ExpressionGraphPtr graph) = 0;
};

typedef std::shared_ptr<OptimizerBase> OptimizerBasePtr;

class Sgd : public OptimizerBase {
  public:
    Sgd(float eta=0.01) : eta_(eta) {}

    void update(ExpressionGraphPtr graph) {
      graph->backprop();

      for(auto& param : graph->params())
        Element(_1 -= eta_ * _2,
                param->val(), param->grad());
    }

  private:
    float eta_;
};

// @TODO: Add serialization for historic gradients and parameters
class Adagrad : public OptimizerBase {
  public:
    Adagrad(float eta=0.01, float eps=1e-8)
    : eta_(eta), eps_(eps),
      alloc_(newTensorAllocator<DeviceGPU>())
    {}

    void update(ExpressionGraphPtr graph) {
      graph->backprop();

      if(!gt_) {
        int totalSize = graph->params().totalSize();
        alloc_->reserveExact(totalSize);
        alloc_->allocate(gt_, {1, totalSize});
        gt_->set(0);
      }

      Tensor pv = graph->params().vals();
      Tensor pg = graph->params().grads();

      Element(_1 += (_2 * _2),
              gt_, pg);

      Element(_1 -= (eta_ / (Sqrt(_2) + eps_)) * _3,
              pv, gt_, pg);
    }

  private:
    float eta_;
    float eps_;
    TensorAllocator alloc_;
    Tensor gt_;
};


// @TODO: Add serialization for historic gradients and parameters
// https://arxiv.org/pdf/1412.6980v8.pdf
class Adam : public OptimizerBase {
  public:
    Adam(float eta=0.001, float beta1=0.9, float beta2=0.999, float eps=1e-8)
    : eta_(eta), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0),
      mtAlloc_(newTensorAllocator<DeviceGPU>()),
      vtAlloc_(newTensorAllocator<DeviceGPU>())
    {}

    void update(ExpressionGraphPtr graph) {
      graph->backprop();

      if(!mt_) {
        int totalSize = graph->params().totalSize();
        mtAlloc_->reserveExact(totalSize);
        mtAlloc_->allocate(mt_, {1, totalSize});
        mt_->set(0);

        vtAlloc_->reserveExact(totalSize);
        vtAlloc_->allocate(vt_, {1, totalSize});
        vt_->set(0);
      }

      t_++;
      float denom1 = 1 - pow(beta1_, t_);
      float denom2 = 1 - pow(beta2_, t_);

      grad_drop(graph, 0.99);

      Tensor pv = graph->params().vals();
      Tensor pg = graph->params().grads();

      //clip(pg);

      Element(_1 = (beta1_ * _1) + ((1 - beta1_) * _2),
              mt_, pg);
      Element(_1 = (beta2_ * _1) + ((1 - beta2_) * (_2 * _2)),
              vt_, pg);

      Element(_1 -= eta_ * (_2 / denom1) / (Sqrt(_3 / denom2) + eps_),
              pv, mt_, vt_);
    }

  private:
    float eta_;
    float beta1_;
    float beta2_;
    float eps_;
    size_t t_;

    TensorAllocator mtAlloc_;
    Tensor mt_;
    TensorAllocator vtAlloc_;
    Tensor vt_;
};

template <class Algorithm, typename ...Args>
OptimizerBasePtr Optimizer(Args&& ...args) {
  return OptimizerBasePtr(new Algorithm(args...));
}

}
