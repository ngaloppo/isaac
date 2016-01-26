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
template<class FUN>
inline void traverse(expression_tree const & tree, size_t root, FUN const & fun)
{
  expression_tree::node const & node = tree[root];
  if (node.type==COMPOSITE_OPERATOR_TYPE){
    traverse(tree, node.binary_operator.lhs, fun);
    traverse(tree, node.binary_operator.rhs, fun);
  }
  if (node.type != INVALID_SUBTYPE)
    fun(root);
}

template<class FUN>
inline void traverse(expression_tree const & tree, FUN const & fun)
{ return traverse(tree, tree.root(), fun); }


//Extract symbolic types
template<class T>
inline void extract(expression_tree const & tree, symbols_table const & table,
                    size_t root, std::set<std::string>& processed, std::vector<T*>& result)
{
  auto extract_impl = [&](size_t index)
  {
    symbols_table::const_iterator it = table.find(index);
    if(it!=table.end())
    {
      T* obj = dynamic_cast<T*>(&*it->second);
      if(obj && processed.insert(obj->process("#name")).second)
        result.push_back(obj);
    }
  };
  traverse(tree, root, extract_impl);
}

template<class T>
inline std::vector<T*> extract(expression_tree const & tree, symbols_table const & table, std::vector<size_t> roots)
{
  std::vector<T*> result;
  std::set<std::string> processed;
  for(size_t root: roots)
     extract(tree, table, root, processed, result);
  return result;
}

template<class T>
inline std::vector<T*> extract(expression_tree const & tree, symbols_table const & table, size_t root)
{
  return extract<T>(tree, table, std::vector<size_t>{root});
}

template<class T>
inline std::vector<T*> extract(expression_tree const & tree, symbols_table const & table)
{
  return extract<T>(tree, table, tree.root());
}

// Filter nodes
std::vector<size_t> find(expression_tree const & tree, size_t root, std::function<bool (expression_tree::node const &)> const & pred);
std::vector<size_t> find(expression_tree const & tree, std::function<bool (expression_tree::node const &)> const & pred);

// Hash
std::string hash(expression_tree const & tree);

//Set arguments
void set_arguments(expression_tree const & tree, driver::Kernel & kernel, unsigned int & current_arg, fusion_policy_t fusion_policy);

//Symbolize
symbols_table symbolize(fusion_policy_t fusion_policy, isaac::expression_tree const & expression);

}
}


#endif
