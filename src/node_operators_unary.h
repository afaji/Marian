#pragma once

#include "node.h"
#include "tensor_operators.h"

namespace marian {

struct UnaryNode : public DifferentiableNode {
    Expr a_;

    template <typename ...Args>
    UnaryNode(Expr a, Args ...args)
    : DifferentiableNode(a->graph(),
           keywords::shape=a->shape(), //@TODO: Check keywords?
           keywords::no_inference=a->skipped_inference() || keywords::Get(keywords::no_inference, false, args...),
           keywords::no_training=a->skipped_training() || keywords::Get(keywords::no_training, false, args...),
           args...),
        a_(a)
    {
        remove_children_from_top_nodes();
    }

    ~UnaryNode() {}

    void remove_children_from_top_nodes();

    void backward_debug(Float delta) {
      using namespace std;

      cerr << "UnaryNode::" << typeid(*this).name() << "::backward_numeric()" << endl;

	  std::vector<float> preCalcGradA, diffGradA, numericalGradA;
	  preCalcGradA << a_->grad();
	  //output("preCalcGradA", preCalcGradA);

	  // use df/dx to calc grad
	  backward();
	  cerr << "orig a_->grad()=" << a_->grad().Debug() << endl;
	  diffGradA << a_->grad();

	  a_->grad().set(preCalcGradA);

	  calc_numeric_grad(delta, a_->val(), a_->grad());
	  cerr << "numerical a_->grad()=" << a_->grad().Debug() << endl;

	  numericalGradA << a_->grad();

	  outputL2Norm("", diffGradA, numericalGradA);

	  // reset to diff grad
	  a_->grad().set(diffGradA);
    }

};

struct SigmoidNode : public UnaryNode {
  template <typename ...Args>
  SigmoidNode(Args ...args)
  : UnaryNode(args...) {  }

  void forward() {
    Element(_1 = Sigma(_2),
            val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * _3 * (1.0f - _3),
            a_->grad(), adj_, val_);
  }

  void check() {

  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("logit")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct TanhNode : public UnaryNode {
  template <typename ...Args>
  TanhNode(Args ...args)
  : UnaryNode(args...) { }

  void forward() {
    Element(_1 = Tanh(_2),
            val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * (1.0f - (_3 * _3)),
            a_->grad(), adj_, val_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("tanh")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

/**
 * Represents a <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">rectified linear</a> node
 *        in an expression graph.
 *
 * This node implements the <a href="https://en.wikipedia.org/wiki/Activation_function">activation function</a>
 *        \f$f(x) = \max(0, x)\f$ and its derivative:
 * 
 \f[
 f^\prime(x) = 
  \begin{cases} 
   0 & \text{if } x \leq 0 \\
   1 & \text{if } x > 0
  \end{cases}
\f]
 */
struct ReLUNode : public UnaryNode {
  template <typename ...Args>
  ReLUNode(Args ...args)
  : UnaryNode(args...) { }

  void forward() {
    Element(_1 = ReLU(_2),
            val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * ReLUback(_3),
            a_->grad(), adj_, a_->val());
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("ReLU")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

/** 
 * @brief Represents a <a href="https://en.wikipedia.org/wiki/Dropout_(neural_networks)">dropout</a> node 
 *        in an expression graph.
 * 
 * @see \cite dropout
 * @see \cite cudnn
 */
struct DropoutNode : public UnaryNode {
  template <typename ...Args>
  DropoutNode(Args ...args)
  : UnaryNode(args...),
    allocated_(false), p_(Get(keywords::value, 0.5)) {}

  ~DropoutNode() {
    if(allocated_)
      CudnnDropoutDestroy(dropDesc_, space_, states_);
 }

  void inference() {
    Element(_1 = _2, val_, a_->val());
  }

  void forward() {
    if(!allocated_) {
        CudnnDropoutPrepare(a_->val(), p_,
                            &dropDesc_,
                            &space_, &spaceSize_,
                            &states_, (size_t)this); // seeding with pointer address
        allocated_ = true;
    }

    CudnnDropoutForward(dropDesc_, space_, spaceSize_,
                        val_, a_->val());
  }

  void backward() {
    CudnnDropoutBackward(dropDesc_, space_, spaceSize_,
                         a_->grad(), adj_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("dropout")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

  private:
    bool allocated_;
    float p_;
    void* states_;
    void* space_;
    size_t spaceSize_;
    cudnnDropoutDescriptor_t dropDesc_;
};

struct SoftmaxNode : public UnaryNode {
  template <typename ...Args>
    SoftmaxNode(Args ...args)
    : UnaryNode(args...) { }

  void forward() {
    CudnnSoftmax(val_, a_->val());
  }

  void backward() {
    // For each row, the Jacobian times vector is given by:
    // J * dy = p .* (dy - avg*1)
    // where avg = p'*dy and p is the softmax output (probabilities).
    //
    // For more information, see sec. 2.5 of the following reference:
    // André F. T. Martins and Ramon Astudillo.
    // "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label
    // Classification." ICML 2016.
    // http://jmlr.org/proceedings/papers/v48/martins16.pdf

    SoftmaxGrad(a_->grad(), adj_, val_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("softmax")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };
};

struct LogSoftmaxNode : public UnaryNode {
  template <typename ...Args>
    LogSoftmaxNode(Args ...args)
    : UnaryNode(args...) { }

  void forward() {
    CudnnLogSoftmax(val_, a_->val());
  }

  void backward() {
    // Based on the description for softmax, we have logsoftmax:
    // J * dy = dy - avg*1
    // where avg = exp(p)'*dy and p is the softmax output (probabilities).
    LogSoftmaxGrad(a_->grad(), adj_, val_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("log-softmax")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };
};


struct ArgmaxNode : public UnaryNode {
  template <typename ...Args>
  ArgmaxNode(Expr a, Args ...args)
    : UnaryNode(a, keywords::shape=newShape(a), args...) { }

  void forward() {
    // B = softmax(A).
    //Argmax(&val_, &a_->val());
  }

  void backward() {
  }

  Shape newShape(Expr a) {
    Shape shape = a->shape();
    shape[1] = 1;
    return shape;
  }


  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label="
      << label("argmax") << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct LogNode : public UnaryNode {
  template <typename ...Args>
  LogNode(Args ...args)
  : UnaryNode(args...) {}

  void forward() {
    Element(_1 = Log(_2), val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * (1.f / _3),
            a_->grad(), adj_, a_->val());
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label="
      << label("log") << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct ExpNode : public UnaryNode {
  template <typename ...Args>
    ExpNode(Args ...args)
    : UnaryNode(args...) { }

  void forward() {
    Element(_1 = Exp(_2), val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * Exp(_3),
            a_->grad(), adj_, a_->val());
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("exp")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct NegNode : public UnaryNode {
  template <typename ...Args>
  NegNode(Args ...args)
  : UnaryNode(args...) { }

  void forward() {
    Element(_1 = -_2, val_, a_->val());
  }

  void backward() {
    Element(_1 += -_2, a_->grad(), adj_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label="
      << label("-") << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};


}
