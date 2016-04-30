#ifndef ISAAC_SYMBOLIC_ENGINE_PROCESS
#define ISAAC_SYMBOLIC_ENGINE_PROCESS

#include <functional>
#include <typeinfo>
#include "isaac/expression.h"
#include "isaac/jit/syntax/engine/object.h"

namespace isaac
{
namespace symbolic
{

//Extract symbolic types
template<class T>
inline void extract(expression const & tree, symbols_table const & table,
                    size_t root, std::set<std::string>& processed, std::vector<T*>& result, bool array_recurse = true)
{
  auto extract_impl = [&](size_t index, size_t)
  {
    symbols_table::const_iterator it = table.find(index);
    if(it!=table.end())
    {
      T* obj = dynamic_cast<T*>(&*it->second);
      if(obj && processed.insert(obj->process("#name")).second)
        result.push_back(obj);
    }
  };
  auto recurse = [&](size_t index){ return array_recurse?true:dynamic_cast<index_modifier*>(&*table.at(index))==0;};
  traverse_dfs(tree, root, extract_impl, recurse);
}

template<class T>
inline std::vector<T*> extract(expression const & tree, symbols_table const & table, std::vector<size_t> roots, bool array_recurse = true)
{
  std::vector<T*> result;
  std::set<std::string> processed;
  for(size_t root: roots)
     extract(tree, table, root, processed, result, array_recurse);
  return result;
}

template<class T>
inline std::vector<T*> extract(expression const & tree, symbols_table const & table, size_t root, bool array_recurse = true)
{
  return extract<T>(tree, table, std::vector<size_t>{root}, array_recurse);
}

template<class T>
inline std::vector<T*> extract(expression const & tree, symbols_table const & table)
{
  return extract<T>(tree, table, tree.root());
}

// Filter nodes
std::vector<size_t> find(expression const & tree, size_t root, std::function<bool (expression::node const &)> const & pred);
std::vector<size_t> find(expression const & tree, std::function<bool (expression::node const &)> const & pred);

std::vector<size_t> assignments(expression const & tree);
std::vector<size_t> lhs_of(expression const & tree, std::vector<size_t> const & in);
std::vector<size_t> rhs_of(expression const & tree, std::vector<size_t> const & in);

// Hash
std::string hash(expression const & tree);

//Set arguments
void set_arguments(expression const & tree, driver::Kernel & kernel, unsigned int & current_arg);

//Symbolize
symbols_table symbolize(expression const & expression);

}
}


#endif
