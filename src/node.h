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

#include <memory>

#include "keywords.h"
#include "tensor.h"

namespace marian {

class ExpressionGraph;
typedef std::shared_ptr<ExpressionGraph> ExpressionGraphPtr;


class ExpressionGraphNode : public keywords::Keywords {

  public:

	template <typename ...Args>
	ExpressionGraphNode(ExpressionGraphPtr graph, Args ...args)
	 : Keywords(args...),
	   graph_(graph),
	   shape_(Get(keywords::shape, {1, 1})),
	   givenShape_(shape_),
	   name_("none")
	{}

	virtual ~ExpressionGraphNode() {}
   
    virtual ExpressionGraphPtr graph();   
   
    virtual void inference() = 0;
        
    virtual std::string graphviz() = 0;

    virtual void allocate(size_t batchSize);
    
    void set_name(const std::string& name);

    const std::string &name() const;
    
    virtual const std::string label(const std::string& type);


    
    virtual const Shape& shape();

    virtual Tensor val();

    
  protected:
  
    ExpressionGraphPtr graph_;
    Shape shape_;
    const Shape givenShape_;
    std::string name_;
    Tensor val_;

};

class DifferentiableNode : public ExpressionGraphNode,
                           public std::enable_shared_from_this<DifferentiableNode> {

  public:

	template <typename ...Args>
	DifferentiableNode(ExpressionGraphPtr graph, Args ...args)
	 : ExpressionGraphNode(graph, args...)
	{}

	virtual ~DifferentiableNode() {}
    
    /**
     * @brief In the context of
     *    <a href="https://justindomke.wordpress.com/2009/03/24/a-simple-explanation-of-reverse-mode-automatic-differentiation/">reverse mode</a>
     *    <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">algorithmic differentiation</a> over an expression graph,
     *    performs forward calculation
     *    for the expression subgraph rooted at this element (aka chainable element \f$i\f$).
     *
     * If this object represents the result of the <em>i</em>-th function in an expression graph,
     * then formally, this method calculates \f$w_i\f$.
     */
    virtual void forward() { }

    /**
     * @brief In the context of
     *    <a href="https://justindomke.wordpress.com/2009/03/24/a-simple-explanation-of-reverse-mode-automatic-differentiation/">reverse mode</a>
     *    <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">algorithmic differentiation</a> over an expression graph,
     *    performs <a href="https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation">reverse accumulation</a>
     *    for the expression subgraph rooted at this element (aka chainable element \f$i\f$).
     *
     * If this object represents the result of the <em>i</em>-th function in an expression graph,
     * then formally, this method calculates \f$\bar{w}_i = \frac{\partial y}{\partial w_i}\f$.
     */
    virtual void backward() { }
    virtual void backward_debug(Float delta) { }

	
	virtual void inference() { 
		forward();
	}

    virtual void skip_inference() { skipInference_ = true; }
    virtual bool skipped_inference() { return skipInference_; }

    virtual void skip_training();

    virtual bool skipped_training() { return skipTraining_; }
    
	virtual void check() { }


    virtual Tensor grad() {
      return adj_;
    };

    virtual void init_dependent() {
      if(adj_) {
        adj_.set(1);
      }
      else {
        adj_.allocate(shape_, 1);
      }
    }

    virtual void set_zero_adjoint() {
      if(adj_) {
        adj_.set(0);
      }
      else {
        adj_.allocate(shape_, 0);
      }
    }
    
  protected:

    Tensor adj_;
    
	bool skipInference_;
    bool skipTraining_;

    void calc_numeric_grad(
              Float delta,
              Tensor input,
              Tensor grad
              );

    void outputL2Norm(const std::string &str, const std::vector<float> &x, const std::vector<float> &y) const;
    float L2Norm(const std::vector<float> &vec) const;

};

/** @brief Defines a convenience type to represent a shared pointer to a DifferentiableNode object. */
typedef std::shared_ptr<DifferentiableNode> Expr;

/**
 * @brief Defines a convenience type to represent an ordered collection items.
 *
 * Conceptually, the items in this collection are pointers to nodes in an expression graph.
 *
 * Naumann (2012) uses "tape" to refer to this data structure.
 * -- The Art of Differentiating Computer Programs: An Introduction to Algorithmic Differentiation, Naumann (2012)
 */
typedef std::vector<Expr> Tape;

}
