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

#include "expression_operators.h"
#include "node_operators.h"

namespace marian {

Expr training(Expr a) {
  a->skip_inference();
  return a;
}

Expr inference(Expr a) {
  a->skip_training();
  return a;
}

Expr name(Expr a, const std::string& name) {
  a->set_name(name);
  a->graph()->add_named_node(a, name);
  return a;
}

Expr logit(Expr a) {
  return Expression<SigmoidNode>(a);
}

Expr tanh(Expr a) {
  return Expression<TanhNode>(a);
}

Expr relu(Expr a) {
  return Expression<ReLUNode>(a);
}

Expr log(Expr a) {
  return Expression<LogNode>(a);
};

Expr exp(Expr a) {
  return Expression<ExpNode>(a);
};

Expr operator-(Expr a) {
  return Expression<NegNode>(a);
};

Expr softmax(Expr a) {
  return Expression<SoftmaxNode>(a);
}

Expr logsoftmax(Expr a) {
  return Expression<LogSoftmaxNode>(a);
}

Expr argmax(Expr a) {
  return Expression<ArgmaxNode>(a);
}

/*********************************************************/

Expr operator+(Expr a, Expr b) {
  return Expression<PlusNode>(a, b);
}

Expr operator-(Expr a, Expr b) {
  return Expression<MinusNode>(a, b);
}

Expr operator*(Expr a, Expr b) {
  return Expression<MultNode>(a, b);
}

Expr operator/(Expr a, Expr b) {
  return Expression<DivNode>(a, b);
}

Expr dot(Expr a, Expr b) {
  return Expression<DotNode>(a, b);
}

Expr reluplus(Expr a, Expr b) {
  return Expression<ReLUPlusNode>(a, b);
}

Expr cross_entropy(Expr a, Expr b) {
  return Expression<CrossEntropyNode>(a, b);
}

}
