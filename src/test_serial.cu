#include <iostream>
#include <string>
#include <fstream>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "marian.h"

using namespace marian;
using namespace keywords;


void save(const std::string& fileName, const ExpressionGraph& graph) {
  std::ofstream ofs(fileName.c_str());
  boost::archive::text_oarchive oa(ofs);
  oa.template register_type< TensorImpl<float> >();
  oa << graph;
}

void load(const std::string& fileName, ExpressionGraph& graph) {
  std::ifstream ifs(fileName.c_str());
  boost::archive::text_iarchive ia(ifs);
  ia.template register_type< TensorImpl<float> >();
  ia >> graph;
}

int main(int argc, char** argv) {
  ExpressionGraph g;

  const size_t n = 2;
  const size_t m = 5;

  Expr x = named(g.input(shape={n, m}), "x");
  Expr a = named(g.param(shape={m, n}, init=normal()), "a");
  Expr y = named(dot(x, a), "y");

  Tensor xt({n, m}, 3.0f);
  x = xt;
  std::cerr << "x: " << g["x"].val().Debug() << std::endl;

  save("tempfile.txt", g);

  g.forward(1);

  std::cerr << "a: " << g["a"].val().Debug() << std::endl;
  std::cerr << "y: " << g["y"].val().Debug() << std::endl;

  ExpressionGraph g2;
  load("tempfile.txt", g2);

  /*std::cerr << "a2: " << g2["a"].val().Debug() << std::endl;*/
  /*std::cerr << "y2: " << g2["y"].val().Debug() << std::endl;*/

  return 0;
}
