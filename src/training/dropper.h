#pragma once
#include <memory>
#include <utility>
#include "kernels/tensor_operators.h"
#include "kernels/cuda_helpers.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <curand_kernel.h>
#include <curand.h>
#include "training/sparse_tensor.h"
//COLWISE: set true to enable column-wise dropping
//MIN-DROP: set true to use min as the quantization center. Otherwise it'll use mean.
#define COLWISE true 
#define MIN_DROP true
//set BIT to 1,2, or 32
#define BIT 2 

namespace marian {
  __global__ void grad_drop(float* data, float* tmp, float* errors, float cut_off, int max_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= max_size)
      return;
  if (std::abs(data[idx])  <= cut_off){
      errors[idx] = data[idx];
      data[idx] = 0;
      tmp[idx] = 0;
  }else{
      errors[idx] = 0;
      tmp[idx] = 1;
  }
}

__global__ void column_wise_quantize(float* input, float* temp_d, float* error, 
              const float T, const size_t lda, const size_t n) {
    __shared__ float sdata[600];
    __shared__ float smin[600];
    __shared__ int scount[600];

    
    float per_block_results;
    float x = 0.0;
    int counter = 0;
    float minimum = 999999999.99;

    float* p = &input[blockIdx.x * lda];
    float* er = &error[blockIdx.x * lda];
    float* tmp = &temp_d[blockIdx.x * lda];
    // Accumulate per thread partial sum
    for(int i=threadIdx.x; i < lda; i += blockDim.x) {
        if (blockIdx.x* lda + i >= n ) printf("%d vs %d\n", blockIdx.x* lda + i, n);
        if (std::abs(p[i]) > T) {
            x += std::abs(p[i]);
            if (minimum > std::abs(p[i]))
                minimum = std::abs(p[i]);
            counter++;
        }
        tmp[i] = 0;
    }

    sdata[threadIdx.x] = x;
    scount[threadIdx.x] = counter;
    smin[threadIdx.x] = minimum;

    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
            scount[threadIdx.x] += scount[threadIdx.x + offset];
            
            if (smin[threadIdx.x] > smin[threadIdx.x + offset])
                smin[threadIdx.x] = smin[threadIdx.x + offset];
        }
        __syncthreads();
    }
    
    if (scount[0] == 0)
        return;
    
    if (MIN_DROP)
        per_block_results = smin[0];
    else
        per_block_results = sdata[0] / (float) scount[0];

    __syncthreads();
    //avg. is obtained. Now replace all:
    for(int i=threadIdx.x; i < lda; i += blockDim.x) {
        if (std::abs(p[i]) <= T) {
            er[i] = p[i];
            p[i] = 0;
        } else {
            int sign = (p[i]>0)?1:-1;
            float replace_to = per_block_results * sign;
            er[i] = p[i] - replace_to;
            p[i] = replace_to;
            tmp[i] = 1;
        }
    }
}

__global__ void grad_drop_quantized(float* data, float* tmp, float* errors, float min_val, float bucket_size, int max_bucket_id, int max_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= max_size)
        return;
    if (std::abs(data[idx])  <= min_val){
        errors[idx] = data[idx];
        data[idx] = 0;
        tmp[idx] = 0;
    }else{
        int sign = (data[idx]<0)?-1:1; 
        int bucket_id = (int)((std::abs(data[idx]) - min_val)/bucket_size);
        if (bucket_id > max_bucket_id)
            bucket_id = max_bucket_id;
        float replace_to = (bucket_id * bucket_size + min_val) * sign;

        errors[idx] = data[idx] - replace_to;
        data[idx] = replace_to;
        tmp[idx] = 1;
    }
}

__global__ void grad_drop_quantized_mean(float* data, float* tmp, float* errors, float min_val, float bucket_size, int max_bucket_id, float mean1, float mean2, int max_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= max_size)
        return;
    if (std::abs(data[idx])  <= min_val){
        errors[idx] = data[idx];
        data[idx] = 0;
        tmp[idx] = 0;
    }else{
        int sign = (data[idx]<0)?-1:1; 
        int bucket_id = (int)((std::abs(data[idx]) - min_val)/bucket_size);
        if (bucket_id > max_bucket_id)
            bucket_id = max_bucket_id;
        float replace_to = mean1;
        if (bucket_id == 1)
            replace_to = mean2;
        replace_to *= sign;

        errors[idx] = data[idx] - replace_to;
        data[idx] = replace_to;
        tmp[idx] = 1;
    }
}

__global__ void grad_add_error(float* data, float* errors, int max_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= max_size)
        return;
    data[idx] += errors[idx];
}

__global__ void full_abs(float* data, int max_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= max_size)
        return;
    data[idx] = abs(data[idx]);
}

__global__ void buildIndices(float* denseData, float* denseSum, float* sparseData, int* sparseIndices,  int denseSize){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= denseSize)
        return;
    int t_id = round(denseSum[idx]);
    if (t_id <= 0){
        return;
  }

  if (idx == 0 &&  t_id > 0){
      sparseIndices[ t_id - 1] = idx;
      sparseData[ t_id - 1] = denseData[idx];
  }
  else if (idx > 0 && t_id > round(denseSum[idx-1]) ){
      sparseIndices[ t_id - 1 ] = idx;
      sparseData[ t_id  - 1] = denseData[idx];
  }
}

__global__ void randomSampling(float* originalData, float* data, int size, int scale, int fullSize){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size)
        return;
    data[idx] = abs(originalData[ idx * scale ] ); 
}

__global__ void locate(float* data, float to_locate, int size, int* result){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    if (data[idx] <= to_locate && (idx == size - 1 || data[idx+1] > to_locate)) 
        *result = idx;
}

class GradientDropBase{

    float* feedback;
    float* temp_d;
    float cut_off;
    int step;
    int _device;

    //a helper. Return i-th element from a GPU stored array
    float get(float* data, int i){
      float res;
      cudaMemcpy(&res, data + i, sizeof(float), cudaMemcpyDeviceToHost);
      return res;
    }

    /*
    algorithm for gradient dropping. 
    
    After executing this function, 100*rate% of lowest absolute value in data will be 0.
    */
    void grad_drop_do(float* data, float* errors, float* tmp, int row_size, int col_size, float rate){
        int len = row_size * col_size;
        int threads = 512;
        int blocks = 1 + len/threads;
        cudaSetDevice(_device);

        //step 1: add the error feedback
        grad_add_error<<<blocks, threads>>>(data, errors, len);

        //step 2: get the T from a sample
        int sortSize = min(100000, len);
        int blocksSample = 1 + sortSize/threads;
        randomSampling<<<blocksSample, threads>>>(data, tmp, sortSize, len / sortSize, len);

        thrust::device_ptr<float> dev_data_ptr(tmp);
        thrust::sort(dev_data_ptr, dev_data_ptr + sortSize );
        int T_index = max(0, (int) (sortSize * rate - 1));
        float T = get(tmp, T_index);

        //step 3: drop
	//CHANGE THIS: only works with 1,2, and 32
        int bit = BIT;
        if (bit == 32){
            grad_drop<<<blocks, threads>>>(data, tmp, errors, T , len);
            return;
        }
          
        if (step == 0) std::cout<<"USING FAKE QUANTIZATION.. bit: "<<bit<<std::endl;
        bit--;
	 //Quantize!
        //get bucket information
        long long bucket_count = (1<<bit);
        float min_val = get(tmp, T_index), max_val = get(tmp, sortSize - 1);
        float range = max_val - min_val;
        float bucket_size = range / bucket_count;
        float mean1 = min_val;
        float mean2 = min_val + bucket_size;

        if (COLWISE && col_size != 1){
            if (step == 0) std::cout<<"COLUMN WISE.."<<std::endl;
 	    column_wise_quantize<<<col_size, 512>>>(data, tmp, errors, min_val, row_size, len);
            return;
        }

        if (MIN_DROP){
            if (step == 0) std::cout<<"MIN-BASED QUANTIZATION.."<<std::endl;
            grad_drop_quantized<<<blocks, threads>>>(data, tmp, errors, min_val, bucket_size, bucket_count - 1, len);
            return;
        }
        if (step == 0) std::cout<<"AVG-BASED QUANTIZATION.."<<std::endl;
        int* result;
        int idx;
        cudaMalloc(&result, sizeof(int));
        locate<<<blocks, threads>>>(tmp, min_val + bucket_size, sortSize, result);
        cudaMemcpy(&idx, result, sizeof(int), cudaMemcpyDeviceToHost); 
        idx++;
        if (idx > T_index && idx <= sortSize) mean1 = thrust::reduce(dev_data_ptr + T_index, dev_data_ptr + idx) / (idx - T_index);
        if (idx > T_index && idx < sortSize) mean2 = thrust::reduce(dev_data_ptr + idx, dev_data_ptr + sortSize) / (sortSize - idx);
        
        cudaFree(result);
        
        grad_drop_quantized_mean<<<blocks, threads>>>(data, tmp, errors, min_val, bucket_size, bucket_count - 1, mean1, mean2, len);
    }

//wrapper
void grad_drop_do(Tensor t, float rate){
  grad_drop_do(t->data(), feedback, temp_d, t->size(), 1, rate);
}

std::vector<std::pair<int,int> > shape_vec;
std::vector<std::pair<std::pair<int,int>, int > > shape_vec_size;

public:

    void dropGraph(Tensor t, SparseTensor destination, double rate, std::vector<std::pair<int,int> > layer_sizes = {}) {
        
        cudaSetDevice(t->getDevice());
        if(!feedback){
            _device = t->getDevice();
            cudaMalloc(&feedback, sizeof(float) * t->size());
            cudaMalloc(&temp_d, sizeof(float) * t->size());
            cudaMemset(feedback, 0, sizeof(float) * t->size());
            cudaMemset(temp_d, 0, sizeof(float) * t->size());

            step = 0;
            for (auto c: layer_sizes)
            	shape_vec.push_back(c);

            int tot_size = 0;
            for (auto& shape: shape_vec){
                std::pair<std::pair<int,int>, int > tmp;
                tmp.first = shape;
                tmp.second = tot_size;
                shape_vec_size.push_back(tmp);
                tot_size += shape.second * shape.first;    
            }
        }
	//if col-wise drop is disabled OR layer sizes info not provided, drop globally
        if (COLWISE == false || shape_vec.size() == 0)
            grad_drop_do( t->data(), feedback, temp_d, t->size(), 1, rate);
        else
        {
	    
            for (auto &shape: shape_vec_size){
                        int offset = shape.second;
                        grad_drop_do( t->data() + offset, feedback + offset, temp_d + offset,  shape.first.first ,shape.first.second, rate);
            }
        }
        if (rate < 0.9)
            return;

        thrust::device_ptr<float> mask_ptr(temp_d);
        int denseSize = t->size();
        thrust::inclusive_scan(mask_ptr, mask_ptr + denseSize , mask_ptr);
        float sparseSize;
        cudaMemcpy(&sparseSize, temp_d + denseSize - 1, sizeof(float), cudaMemcpyDeviceToHost);
            //convert result of exscan to indices.
        int threads = 512;
        int blocks = 1 + denseSize/threads;
        cudaSetDevice(t->getDevice());
        buildIndices<<<blocks, threads>>>(t->data(), temp_d, destination->data(), destination->indices(),  denseSize);
        destination->setSize(sparseSize);

        cudaStreamSynchronize(0);
	step++;
     }
 };


 typedef Ptr<GradientDropBase> GradientDrop;

}
