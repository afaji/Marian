#include "node_operators_unary.h"
#include "expression_graph.h"

namespace marian {
  void UnaryNode::remove_children_from_top_nodes() {
    graph_->remove_top_node(a_);
  }
}
