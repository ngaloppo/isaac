/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#include "isaac/driver/context.h"
#include "isaac/driver/kernel.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{
namespace symbolic
{

// Filter nodes
std::vector<size_t> find(expression const & tree, size_t root, std::function<bool (expression::node const &)> const & pred)
{
  std::vector<size_t> result;
  auto fun = [&](size_t index, size_t) { if(pred(tree[index])) result.push_back(index); };
  traverse_dfs(tree, root, fun);
  return result;
}

std::vector<size_t> find(expression const & tree, std::function<bool (expression::node const &)> const & pred)
{
  return find(tree, tree.root(), pred);
}

std::vector<size_t> assignments(expression const & tree)
{
  return find(tree, [&](expression::node const & node)
            {return node.type==COMPOSITE_OPERATOR_TYPE && node.binary_operator.op.type==ASSIGN_TYPE;}
          );
}

std::vector<size_t> lhs_of(expression const & tree, std::vector<size_t> const & in)
{
  std::vector<size_t> result;
  for(size_t idx: in)
    result.push_back(tree[idx].binary_operator.lhs);
  return result;
}

std::vector<size_t> rhs_of(expression const & tree, std::vector<size_t> const & in)
{
  std::vector<size_t> result;
  for(size_t idx: in)
    result.push_back(tree[idx].binary_operator.rhs);
  return result;
}


// Hash
std::string hash(expression const & tree)
{
  char program_name[256];
  char* ptr = program_name;
  auto hash_impl = [&](size_t idx, size_t)
  {
    expression::node const & node = tree[idx];
    if(node.type==DENSE_ARRAY_TYPE)
    {
      for(size_t i = 0 ; i < node.shape.size() ; ++i)
        *ptr++= node.shape[i]>1?'n':'1';
      if(node.ld[0]>1) *ptr++= 's';
      *ptr++=(char)node.dtype;
      tools::fast_append(ptr, node.id);
    }
    else if(node.type==COMPOSITE_OPERATOR_TYPE){
      tools::fast_append(ptr,node.binary_operator.op.family);
      tools::fast_append(ptr,node.binary_operator.op.type);
    }
  };
  traverse_dfs(tree, hash_impl);
  *ptr='\0';
  return std::string(program_name);
}

//Set arguments
void set_arguments(expression const & tree, driver::Kernel & kernel, unsigned int & current_arg)
{
  driver::backend_type backend = tree.context().backend();
  std::set<size_t> already_set;
  //set_arguments_impl
  auto set_arguments_impl = [&](size_t index, size_t)
  {
    expression::node const & node = tree[index];
    if(node.type==VALUE_SCALAR_TYPE)
      kernel.setArg(current_arg++,scalar(node.value,node.dtype));
    else if(node.type==DENSE_ARRAY_TYPE)
    {
      if (already_set.insert(index).second)
      {
          if(backend==driver::OPENCL)
            kernel.setArg(current_arg++, node.array.handle.cl);
          else
            kernel.setArg(current_arg++, node.array.handle.cu);
          kernel.setSizeArg(current_arg++, node.array.start);
          for(size_t i = 0 ; i < node.shape.size() ; i++)
            if(node.shape[i] > 1)
              kernel.setSizeArg(current_arg++, node.ld[i]);
      }
    }
    else if(node.type==COMPOSITE_OPERATOR_TYPE && node.binary_operator.op.type == RESHAPE_TYPE)
    {
      tuple const & new_shape = node.shape;
      int_t current = 1;
      for(size_t i = 1 ; i < new_shape.size() ; ++i){
        current *= new_shape[i-1];
        if(new_shape[i] > 1)
          kernel.setSizeArg(current_arg++, current);
      }

      tuple const & old_shape = tree[node.binary_operator.lhs].shape;
      current = 1;
      for(unsigned int i = 1 ; i < old_shape.size() ; ++i){
        current *= old_shape[i-1];
        if(old_shape[i] > 1)
          kernel.setSizeArg(current_arg++, current);
      }
    }
  };


  //Traverse
  traverse_dfs(tree, set_arguments_impl);
}

//Symbolize
template<class T, class... Args>
std::shared_ptr<object> make_symbolic(Args&&... args)
{
  return std::shared_ptr<object>(new T(std::forward<Args>(args)...));
}

symbols_table symbolize(isaac::expression const & tree)
{
  driver::Context const & context = tree.context();
  symbols_table result;
  //Functor
  auto symbolize_impl = [&](size_t root, size_t)
  {
    expression::node const & node = tree[root];
    std::string dtype = to_string(node.dtype);
    if(node.type==VALUE_SCALAR_TYPE)
      result.insert({root, make_symbolic<host_scalar>(context, dtype, node.id)});
    else if(node.type==DENSE_ARRAY_TYPE){
      result.insert({root, make_symbolic<buffer>(context, dtype, node.id, node.shape, node.ld)});
    }
    else if(node.type==COMPOSITE_OPERATOR_TYPE)
    {
      token op = node.binary_operator.op;
      //Index modifier
      if(op.type==RESHAPE_TYPE)
        result.insert({root, make_symbolic<reshape>(dtype, node.id, root, op, tree, result)});
      else if(op.type==TRANS_TYPE)
        result.insert({root, make_symbolic<trans>(dtype, node.id, root, op, tree, result)});
      else if(op.type==DIAG_VECTOR_TYPE)
        result.insert({root, make_symbolic<diag_vector>(dtype, node.id, root, op, tree, result)});
      //Unary arithmetic
      else if(op.family==UNARY_ARITHMETIC)
        result.insert({root, make_symbolic<unary_arithmetic_node>(node.id, root, op, tree, result)});
      //Binary arithmetic
      else if(op.family==BINARY_ARITHMETIC)
        result.insert({root, make_symbolic<binary_arithmetic_node>(node.id, root, op, tree, result)});
      //1D Reduction
      else if (op.family==REDUCE)
        result.insert({root, make_symbolic<reduce_1d>(node.id, root, op, tree, result)});
      //2D reduction
      else if (op.family==REDUCE_ROWS || op.family==REDUCE_COLUMNS)
        result.insert({root, make_symbolic<reduce_2d>(node.id, root, op, tree, result)});
    }
  };
  //traverse
  traverse_dfs(tree, symbolize_impl);
  return result;
}

}
}
