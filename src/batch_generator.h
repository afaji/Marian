#pragma once

#include <deque>
#include <queue>

#include <boost/timer/timer.hpp>

#include "dataset.h"

namespace marian {

namespace data {

template <class DataSet>
class BatchGenerator {
  public:
    typedef typename DataSet::batch_ptr BatchPtr;

  private:
    std::shared_ptr<DataSet> data_;
    ExampleIterator current_;

    size_t batchSize_;
    size_t maxiBatchSize_;

    std::deque<BatchPtr> bufferedBatches_;
    BatchPtr currentBatch_;

    void fillBatches() {
      auto cmp = [](const ExamplePtr& a, const ExamplePtr& b) {
        return (*a)[0]->size() < (*b)[0]->size();
      };

      std::priority_queue<ExamplePtr, Examples, decltype(cmp)> maxiBatch(cmp);

      while(current_ != data_->end() && maxiBatch.size() < maxiBatchSize_) {
        maxiBatch.push(*current_);
        current_++;
      }

      Examples batchVector;
      while(!maxiBatch.empty()) {
        batchVector.push_back(maxiBatch.top());
        maxiBatch.pop();
        if(batchVector.size() == batchSize_) {
          bufferedBatches_.push_back(data_->toBatch(batchVector));
          batchVector.clear();
        }
      }
      if(!batchVector.empty())
        bufferedBatches_.push_back(data_->toBatch(batchVector));

      std::random_shuffle(bufferedBatches_.begin(), bufferedBatches_.end());
      //std::cerr << "Total: " << total.format(5, "%ws") << std::endl;
    }

  public:
    BatchGenerator(std::shared_ptr<DataSet> data,
                   size_t batchSize=80,
                   size_t maxiBatchNum=20)
    : data_(data),
      batchSize_(batchSize),
      maxiBatchSize_(batchSize * maxiBatchNum),
      current_(data_->begin()) { }

    operator bool() const {
      return !bufferedBatches_.empty();
    }

    BatchPtr next() {
      UTIL_THROW_IF2(bufferedBatches_.empty(),
                     "No batches to fetch");
      currentBatch_ = bufferedBatches_.front();
      bufferedBatches_.pop_front();

      if(bufferedBatches_.empty())
        fillBatches();

      return currentBatch_;
    }

    void prepare(bool shuffle=true) {
      //boost::timer::cpu_timer total;
      if(shuffle)
        data_->shuffle();
      //std::cerr << "shuffle: " << total.format(5, "%ws") << std::endl;
      current_ = data_->begin();
      fillBatches();
    }
};

}

}
