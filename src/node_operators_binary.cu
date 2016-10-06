#include "node_operators_binary.h"
#include "expression_graph.h"

namespace marian {
  void BinaryNode::remove_children_from_top_nodes() {
    graph_->remove_top_node(a_);
    graph_->remove_top_node(b_);
  }
}
