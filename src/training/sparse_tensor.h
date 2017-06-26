#pragma once
#include <memory>
#include "kernels/tensor_operators.h"
#include "kernels/cuda_helpers.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <curand_kernel.h>
#include <curand.h>

namespace marian {
	// @TODO: use inplace Radix Select
	// create actual sparse tensor class. This one is just minimal 
	__global__ void gScatterUpdate(float* denseData, float* sparseData, int* sparseIndices, int denseSize, int sparseSize, int offset){
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= sparseSize)
			return;
		if (sparseIndices[idx] + offset >= 0 && sparseIndices[idx] + offset < denseSize)
			denseData[ sparseIndices[idx] + offset ] = sparseData[idx];
	}

	__global__ void gScatterCopy(float* denseData, float* sparseData, int* sparseIndices, int denseSize, int sparseSize){
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= sparseSize)
			return;
		if (sparseIndices[idx]  >= 0 && sparseIndices[idx] < denseSize)
			sparseData[idx] = denseData[ sparseIndices[idx] ];
	}

	__global__ void gShift(int* indices, int size, int offset){
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size)
			return;
		indices[idx] += offset;
	}

	__global__ void gFindSubtensor(int* indices, int size, int targetStart, int targetEnd, int* resultStart, int* resultEnd){


		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= size)
			return;

		if (indices[idx] >= targetStart && (idx ==0 || indices[idx - 1] < targetStart)){
			resultStart[0] = idx;
		}

		if (indices[idx] < targetEnd && (idx == size - 1 || indices[idx + 1] >= targetEnd))
			resultEnd[0] = idx;
	}


	class SparseTensorBase: public std::enable_shared_from_this<SparseTensorBase> {
		float *data_;
		int *indices_;
		int size_;
		int capacity_;
		size_t device_; 


		int* d_is_unsorted;
		int* gstart_;
		int *gend_;

	public:
		SparseTensorBase(int capacity, size_t device){
			device_ = device;
			capacity_ = capacity;
			cudaSetDevice(device_);
			cudaMalloc(&data_, sizeof(float) * capacity);
	    	cudaMalloc(&indices_, sizeof(int) * capacity);

	    	cudaMalloc(&gstart_, sizeof(int) * 100);
	    	cudaMalloc(&gend_, sizeof(int) * 100);

		}

		SparseTensorBase(float* data, int* indices, int size, size_t device){
			data_ = data;
			indices_ = indices;
			size_ = size;
			capacity_ = size;
			device_ = device;
		}

		~SparseTensorBase(){
		}

		int capacity(){
			return capacity_;
		}

		int size(){
			return size_;
		}

		float* data(){
			return data_;
		}

		int* indices(){
			return indices_;
		}

		void copyFrom(float* data, int* indices, int size, bool data_only){
			if (capacity_ < size){
				return;
				//NO enough capacity
			}
			size_ = size;
			if(size == 0) return;
			cudaSetDevice(device_);

			cudaMemcpy(data_, data, size * sizeof(float), cudaMemcpyDefault);
			if (!data_only)
				cudaMemcpy(indices_, indices, size * sizeof(int), cudaMemcpyDefault);
			cudaStreamSynchronize(0);
		}

		void copyFrom(std::shared_ptr<SparseTensorBase> t, bool data_only = false){
			copyFrom(t->data(), t->indices(), t->size(), data_only);
		}

		void copyFromDense(Tensor t ){
			cudaSetDevice(device_);	

		}

		size_t getDevice(){
			return device_;
		}

		void setSize(int size){
			size_ = size;
		}

		void shiftIndices(int offset){
			cudaSetDevice(device_);
			int threads = 512;
			int blocks = 1 + size_ /threads;
			gShift<<<blocks, threads>>> (indices_, size_, offset);
		}

		void toDense(Tensor t, int offset){
			cudaSetDevice(device_);
			int threads = 512;
			int blocks = 1 + size_ /threads;
			t->set(0);
			gScatterUpdate<<<blocks, threads>>> (t->data() , data_, indices_, t->size(), size_, offset);
		}

		void scatterUpdate(Tensor t, int offset){
			cudaSetDevice(device_);
			cudaStreamSynchronize(0);
			int threads = 512;
			int blocks = 1 + size_ /threads;
			gScatterUpdate<<<blocks, threads>>> (t->data() , data_, indices_, t->size(), size_ , offset);
		}

		void scatterCopyFrom(Tensor t){
			cudaSetDevice(device_);
			int threads = 512;
			int blocks = 1 + size_ /threads;
			gScatterCopy<<<blocks, threads>>> (t->data() , data_, indices_, t->size(), size_);
			cudaStreamSynchronize(0);
		}

		std::shared_ptr<SparseTensorBase> subtensor(int pos, int size, int idx){
			cudaSetDevice(device_);
			cudaStreamSynchronize(0);
			int* start = gstart_ + idx;
			int* end = gend_ + idx;


			int threads = 512;
			int blocks = 1 + size_ /threads;
			cudaMemset(start, -1, sizeof(int));
			cudaMemset(end, 0, sizeof(int));

			gFindSubtensor<<<blocks, threads>>> (indices_ , size_, pos, pos + size, start, end);
			
			int startOffset;
			int endOffset;
			int tmp_dt;
			cudaMemcpy(&startOffset, start, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&endOffset, end, sizeof(int), cudaMemcpyDeviceToHost);
			
			if (startOffset != -1 && startOffset < size_) cudaMemcpy(&tmp_dt, indices_ + startOffset , sizeof(int), cudaMemcpyDeviceToHost);

			int subtensorSize = max(0, endOffset - startOffset + 1);
			cudaStreamSynchronize(0);
			return std::shared_ptr<SparseTensorBase>( new SparseTensorBase(data_ + startOffset, indices_ + startOffset, subtensorSize, device_) );
		}
	};

	typedef std::shared_ptr<SparseTensorBase> SparseTensor;
}