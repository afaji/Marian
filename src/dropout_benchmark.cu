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

#include <fstream>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "mnist.h"
#include "vocab.h"
#include "tensor_operators.h"
#include "curand.h"

using namespace marian;
using namespace keywords;

int main(int argc, char** argv) {
  
  Tensor a({1000, 1000}, 3);
  Tensor b({1000, 1000});
  Bernoulli dropout(0.2, b.shape());
  
  auto f = [] __device__ (float& r,
                          float a,
                          float b)  {
    return r = a * b;
  };
    
  boost::timer::cpu_timer timer;
  for(int i = 0; i < 1000; ++i)
    Element(f, b, a, a);
  
  std::cerr << timer.format(5, "%ws") << std::endl;
  return 0;
}
