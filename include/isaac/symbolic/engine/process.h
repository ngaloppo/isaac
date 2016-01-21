#ifndef ISAAC_SYMBOLIC_ENGINE_PROCESS
#define ISAAC_SYMBOLIC_ENGINE_PROCESS

#include "isaac/tools/cpp/string.hpp"
#include "isaac/symbolic/expression/expression.h"
#include "isaac/symbolic/engine/binder.h"
#include "isaac/symbolic/engine/object.h"
#include "isaac/array.h"

namespace isaac
{
namespace symbolic
{


//Traverse
namespace detail
{
  template<class FUN>
  inline void traverse(expression_tree const & tree, size_t root, FUN const & fun)
  {
    expression_tree::node const & root_node = tree.data()[root];

    //Lhs:
    if (root_node.lhs.type==COMPOSITE_OPERATOR_TYPE)
      traverse(tree, root_node.lhs.index, fun);
    if (root_node.lhs.type != INVALID_SUBTYPE)
      fun(root, LHS_NODE_TYPE);

    //Rhs:
    if (root_node.rhs.type!=INVALID_SUBTYPE)
    {
      if (root_node.rhs.type==COMPOSITE_OPERATOR_TYPE)
        traverse(tree, root_node.rhs.index, fun);
      if (root_node.rhs.type != INVALID_SUBTYPE)
        fun(root, RHS_NODE_TYPE);
    }

    fun(root, PARENT_NODE_TYPE);
  }
}

template<class FUN>
inline void traverse(expression_tree const & expression, size_t idx, leaf_t leaf, FUN const & fun)
{
  expression_tree::node const & root = expression.data()[idx];
  if(leaf==RHS_NODE_TYPE && root.rhs.type==COMPOSITE_OPERATOR_TYPE)
    detail::traverse(expression, root.rhs.index, fun);
  else if(leaf==LHS_NODE_TYPE && root.lhs.type==COMPOSITE_OPERATOR_TYPE)
    detail::traverse(expression, root.lhs.index, fun);
  else if(leaf==PARENT_NODE_TYPE)
    detail::traverse(expression, idx, fun);
  else
    fun(idx, leaf);
}

template<class FUN>
inline void traverse(expression_tree const & tree, FUN const & fun)
{ return traverse(tree, tree.root(), PARENT_NODE_TYPE, fun); }


//Extract symbolic types
template<class T>
inline void extract(expression_tree const & expression, symbols_table const & symbolic,
                    size_t idx, leaf_t leaf, std::set<std::string>& processed, std::vector<T*>& result)
{
  auto extract_impl = [&](size_t index, leaf_t leaf)
  {
    symbols_table::const_iterator it = symbolic.find({index, leaf});
    if(it!=symbolic.end())
    {
      T* obj = dynamic_cast<T*>(&*it->second);
      if(obj && processed.insert(obj->process("#name")).second)
        result.push_back(obj);
    }
  };
  traverse(expression, idx, leaf, extract_impl);
}

template<class T>
inline std::vector<T*> extract(expression_tree const & expression, symbols_table const & symbolic, std::vector<size_t> idxs, leaf_t leaf = PARENT_NODE_TYPE)
{
  std::vector<T*> result;
  std::set<std::string> processed;
  for(size_t idx: idxs)
     extract(expression, symbolic, idx, leaf, processed, result);
  return result;
}

template<class T>
inline std::vector<T*> extract(expression_tree const & expression, symbols_table const & symbolic, size_t root, leaf_t leaf = PARENT_NODE_TYPE)
{
  return extract<T>(expression, symbolic, std::vector<size_t>{root}, leaf);
}

template<class T>
inline std::vector<T*> extract(expression_tree const & expression, symbols_table const & symbolic)
{
  return extract<T>(expression, symbolic, expression.root(), PARENT_NODE_TYPE);
}

// Filter nodes
std::vector<size_t> filter(expression_tree const & expression, size_t idx, leaf_t leaf, std::function<bool (expression_tree::node const &)> const & pred);
std::vector<size_t> filter(expression_tree const & expression, std::function<bool (expression_tree::node const &)> const & pred);

// Hash
std::string hash(expression_tree const & expression);

//Set arguments
void set_arguments(expression_tree const & expression, driver::Kernel & kernel, unsigned int & current_arg, fusion_policy_t fusion_policy);

//Symbolize
symbols_table symbolize(fusion_policy_t fusion_policy, isaac::expression_tree const & expression);

}
}


#endif
