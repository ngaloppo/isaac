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

#ifndef ISAAC_KERNELS_PARSE_SET_ARGUMENTS_HPP
#define ISAAC_KERNELS_PARSE_SET_ARGUMENTS_HPP

#include <cstring>

#include "isaac/symbolic/expression/expression.h"
#include "isaac/symbolic/engine/binder.h"
#include "isaac/symbolic/engine/object.h"
#include "traverse.hpp"
#include "isaac/array.h"

namespace isaac
{

inline void set_arguments(expression_tree const & expression, driver::Kernel & kernel, unsigned int & current_arg, binding_policy_t binding_policy)
{
  //Create binder
  std::unique_ptr<symbolic_binder> binder;
  if (binding_policy==BIND_SEQUENTIAL)
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

}

#endif
