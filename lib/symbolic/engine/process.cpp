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

#include "isaac/symbolic/engine/process.h"

namespace isaac
{
namespace symbolic
{

// Filter nodes
std::vector<size_t> filter(expression_tree const & expression, size_t idx, leaf_t leaf, std::function<bool (expression_tree::node const &)> const & pred)
{
  std::vector<size_t> result;
  auto fun = [&](size_t index, leaf_t leaf) {  if(leaf==PARENT_NODE_TYPE && pred(expression.data()[index])) result.push_back(index); };
  _traverse(expression, idx, leaf, fun);
  return result;
}

std::vector<size_t> filter(expression_tree const & expression, std::function<bool (expression_tree::node const &)> const & pred)
{
  return filter(expression, expression.root(), PARENT_NODE_TYPE, pred);
}

// Hash
std::string hash(expression_tree const & expression)
{
  char program_name[256];
  char* ptr = program_name;
  bind_independent binder;

  auto hash_leaf = [&](tree_node const & node, bool is_assigned)
  {
    if(node.type==DENSE_ARRAY_TYPE)
    {
        for(int i = 0 ; i < node.array->dim() ; ++i)
          *ptr++= node.array->shape()[i]>1?'n':'1';
        numeric_type dtype = node.array->dtype();
        *ptr++=(char)dtype;
        tools::fast_append(ptr, binder.get(node.array, is_assigned));
    }
  };

  auto hash_impl = [&](size_t idx, leaf_t leaf)
  {
    expression_tree::node const & node = expression.data()[idx];
    if (leaf==LHS_NODE_TYPE && node.lhs.type != COMPOSITE_OPERATOR_TYPE)
      hash_leaf(node.lhs, is_assignment(node.op.type));
    else if (leaf==RHS_NODE_TYPE && node.rhs.type != COMPOSITE_OPERATOR_TYPE)
      hash_leaf(node.rhs, false);
    else if (leaf==PARENT_NODE_TYPE)
      tools::fast_append(ptr,node.op.type);
  };

  _traverse(expression, hash_impl);

  *ptr='\0';

  return std::string(program_name);
}

//Set arguments
void set_arguments(expression_tree const & expression, driver::Kernel & kernel, unsigned int & current_arg, fusion_policy_t fusion_policy)
{
  //Create binder
  std::unique_ptr<symbolic_binder> binder;
  if (fusion_policy==FUSE_SEQUENTIAL)
      binder.reset(new bind_sequential());
  else
      binder.reset(new bind_independent());

  //setLeafArg
  auto set_leaf_arguments = [&](tree_node const & leaf, bool is_assigned)
  {
    if(leaf.type==VALUE_SCALAR_TYPE)
      kernel.setArg(current_arg++,value_scalar(leaf.scalar,leaf.dtype));
    else if(leaf.type==DENSE_ARRAY_TYPE)
    {
      array_base const * array = leaf.array;
      bool is_bound = binder->bind(array, is_assigned);
      if (is_bound)
      {
          kernel.setArg(current_arg++, array->data());
          kernel.setSizeArg(current_arg++, array->start());
          for(int_t i = 0 ; i < array->dim() ; i++)
          {
            if(array->shape()[i] > 1)
              kernel.setSizeArg(current_arg++, array->stride()[i]);
          }
      }
    }
  };

  //set_arguments_impl
  auto set_arguments_impl = [&](size_t root_idx, leaf_t leaf)
  {
    expression_tree::node const & node = expression.data()[root_idx];
    if (leaf==LHS_NODE_TYPE && node.lhs.type != COMPOSITE_OPERATOR_TYPE)
      set_leaf_arguments(node.lhs, is_assignment(node.op.type));
    else if (leaf==RHS_NODE_TYPE && node.rhs.type != COMPOSITE_OPERATOR_TYPE)
      set_leaf_arguments(node.rhs, false);
    if(leaf==PARENT_NODE_TYPE && node.op.type == RESHAPE_TYPE)
    {
      tuple const & new_shape = node.shape;
      int_t current = 1;
      for(unsigned int i = 1 ; i < new_shape.size() ; ++i){
        current *= new_shape[i-1];
        if(new_shape[i] > 1)
          kernel.setSizeArg(current_arg++, current);
      }

      tuple const & old_shape = node.lhs.type==DENSE_ARRAY_TYPE?node.lhs.array->shape():expression.data()[node.lhs.index].shape;
      current = 1;
      for(unsigned int i = 1 ; i < old_shape.size() ; ++i){
        current *= old_shape[i-1];
        if(old_shape[i] > 1)
          kernel.setSizeArg(current_arg++, current);
      }
    }
  };


  //Traverse
  _traverse(expression, set_arguments_impl);
}

//Symbolize
template<class T, class... Args>
std::shared_ptr<object> make_symbolic(Args&&... args)
{
  return std::shared_ptr<object>(new T(std::forward<Args>(args)...));
}

symbols_table symbolize(fusion_policy_t fusion_policy, isaac::expression_tree const & expression)
{
  //binder
  symbols_table result;
  std::unique_ptr<symbolic_binder> binder;
  if (fusion_policy==FUSE_SEQUENTIAL)
      binder.reset(new bind_sequential());
  else
      binder.reset(new bind_independent());

  //make_leaf
  auto symbolize_leaf = [&](std::string const & dtype, tree_node const & leaf, bool is_assigned)
  {
    if(leaf.type==VALUE_SCALAR_TYPE)
      return make_symbolic<host_scalar>(dtype, binder->get());
    else if(leaf.type==DENSE_ARRAY_TYPE)
      return make_symbolic<buffer>(dtype, binder->get(leaf.array, is_assigned), leaf.array->shape());
    else if(leaf.type==PLACEHOLDER_TYPE)
      return make_symbolic<placeholder>(leaf.ph.level);
    else
      throw;
  };

  //make_impl
  auto symbolize_impl = [&](size_t idx, leaf_t leaf)
  {
    symbols_table::key_type key(idx, leaf);
    expression_tree::node const & node = expression.data()[idx];
    std::string dtype = to_string(expression.dtype());


    if (leaf == LHS_NODE_TYPE && node.lhs.type != COMPOSITE_OPERATOR_TYPE)
      result.insert({key, symbolize_leaf(dtype, node.lhs, is_assignment(node.op.type))});
    else if (leaf == RHS_NODE_TYPE && node.rhs.type != COMPOSITE_OPERATOR_TYPE)
      result.insert({key, symbolize_leaf(dtype, node.rhs, false)});
    else if (leaf== PARENT_NODE_TYPE)
    {
      unsigned int id = binder->get();
      //Index modifier
      if (node.op.type==VDIAG_TYPE)
        result.insert({key, make_symbolic<diag_matrix>(dtype, id, idx, result)});
      else if (node.op.type==MATRIX_DIAG_TYPE)
        result.insert({key, make_symbolic<diag_vector>(dtype, id, idx, result)});
      else if (node.op.type==MATRIX_ROW_TYPE)
        result.insert({key, make_symbolic<matrix_row>(dtype, id, idx, result)});
      else if (node.op.type==MATRIX_COLUMN_TYPE)
        result.insert({key, make_symbolic<matrix_column>(dtype, id, idx, result)});
      else if(node.op.type==ACCESS_INDEX_TYPE)
        result.insert({key, make_symbolic<array_access>(dtype, id, idx, result)});
      else if(node.op.type==RESHAPE_TYPE)
        result.insert({key, make_symbolic<reshape>(dtype, id, idx, expression, result)});
      //Unary arithmetic
      else if(node.op.type_family==UNARY)
        result.insert({key, make_symbolic<unary_node>(node.op.type, idx, expression, result)});
      //Binary arithmetic
      else if(node.op.type_family==BINARY)
        result.insert({key, make_symbolic<binary_node>(node.op.type, idx, expression, result)});
      //1D Reduction
      else if (node.op.type_family==REDUCE)
        result.insert({key, make_symbolic<reduce_1d>(dtype, id, idx, node.op)});
      //2D reduction
      else if (node.op.type_family==REDUCE_ROWS || node.op.type_family==REDUCE_COLUMNS)
        result.insert({key, make_symbolic<reduce_2d>(dtype, id, idx, node.op)});
    }
  };

  //traverse
  _traverse(expression, symbolize_impl);

  return result;
}

}
}
