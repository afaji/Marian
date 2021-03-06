#include "node_operators_binary.h"
#include "expression_graph.h"

namespace marian {
  void BinaryNodeOp::remove_children_from_top_nodes() {
    graph_->remove_top_node(a_);
    graph_->remove_top_node(b_);
  }

  void NaryNodeOp::remove_children_from_top_nodes() {
    for(auto child : children_)
      graph_->remove_top_node(child);
  }

}
