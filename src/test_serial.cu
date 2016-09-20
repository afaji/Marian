#include <iostream>
#include <string>
#include <fstream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "marian.h"

using namespace marian;
using namespace keywords;


void save(const std::string& fileName, const ExpressionGraph& graph) {
  std::ofstream ofs(fileName.c_str());
  boost::archive::text_oarchive oa(ofs);
  /*oa.template register_type< TypedSubClass<int> >();*/
  oa << graph;
}

void load(const std::string& fileName, ExpressionGraph& graph) {
  std::ifstream ifs(fileName.c_str());
  boost::archive::text_iarchive ia(ifs);
  /*ia.template register_type< TypedSubClass<int> >();*/
  ia >> graph;
}

int main(int argc, char** argv) {
  ExpressionGraph g;

  const size_t n = 2;
  const size_t m = 5;

  Expr x = g.input(shape={n, m}, name="x");
  Expr a = g.param(shape={m, n}, name="a", init=normal());
  Expr y = dot(x, a);

  Tensor xt({n, m}, 3.0f);
  x = xt;
  std::cerr << "x: " << x.val().Debug() << std::endl;

  save("tempfile.txt", g);

  g.forward(1);

  std::cerr << "a: " << a.val().Debug() << std::endl;
  std::cerr << "y: " << y.val().Debug() << std::endl;

  return 0;
}
