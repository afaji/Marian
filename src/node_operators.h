#pragma once

// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "node.h"
#include "tensor_operators.h"
#include "node_operators_unary.h"
#include "node_operators_binary.h"

namespace marian {

struct InputNode : public DifferentiableNode {
  template <typename ...Args>
  InputNode(ExpressionGraphPtr graph, Args ...args)
  : DifferentiableNode(graph, args...) {
    UTIL_THROW_IF2(!Has(keywords::shape) &&
                   !Has(keywords::lazy_shape),
                   "Data items require shape information");
  }

  ~InputNode() {}

  virtual void setVal(Tensor t)  {
    val_ = t;
    shape_ = t.shape();
    //@todo, shape checking
  }

  void forward() {}
  void backward() {}

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"circle\", label=" << label("input") << ", style=\"filled\", fillcolor=\"lawngreen\"]" << std::endl << std::endl;
    return ss.str();
  }

};


struct ConstantNode : public DifferentiableNode {
  template <typename ...Args>
  ConstantNode(ExpressionGraphPtr graph, Args ...args)
  : DifferentiableNode(graph, args...) {
    UTIL_THROW_IF2(!Has(keywords::shape) &&
                   !Has(keywords::lazy_shape),
                   "Constant items require shape information");
  }

  ~ConstantNode() {}

  void forward() {}
  void backward() {}

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"diamond\", label=" << label("const") << "]" << std::endl << std::endl;
    return ss.str();
  }

};

struct ParamNode : public DifferentiableNode {
  template <typename ...Args>
  ParamNode(ExpressionGraphPtr graph, Args ...args)
  : DifferentiableNode(graph, args...),
    init_(Get(keywords::init, [](Tensor){ })),
    initialized_(false)
  {
    UTIL_THROW_IF2(!Has(keywords::shape) &&
                   !Has(keywords::lazy_shape),
                   "Param items require shape information");
  }

  virtual void setVal(Tensor t)  {
    val_ = t;
    shape_ = t.shape();
    //@todo, shape checking
  };

  ~ParamNode() {}

  virtual void setGrad(Tensor t)  {
    adj_ = t;
    shape_ = t.shape();
    //@todo, shape checking
  };

  void forward() {}
  void backward() {}

  virtual void allocate(size_t batchSize) {
    val_.allocate(shape_);
    if(!initialized_) {
      init_(val_);
      initialized_ = true;
    }
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"hexagon\", label=" << label("param")
      << ", style=\"filled\", fillcolor=\"orangered\"]"
      << std::endl << std::endl;
    return ss.str();
  };


  private:
    std::function<void(Tensor)> init_;
    bool initialized_;
};


/** @brief Defines a convenience type to represent a shared pointer to a DifferentiableNode object. */
typedef std::shared_ptr<InputNode> InputLayer;

/** @brief Defines a convenience type to represent a shared pointer to a DifferentiableNode object. */
typedef std::shared_ptr<ParamNode> ConnectionWeights;


}
