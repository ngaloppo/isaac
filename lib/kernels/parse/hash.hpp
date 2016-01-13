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

#ifndef ISAAC_KERNELS_PARSE_HASH_HPP
#define ISAAC_KERNELS_PARSE_HASH_HPP

#include <cstring>
#include "traverse.hpp"
#include "isaac/array.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{

inline std::string hash(expression_tree const & expression)
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
      hash_leaf(node.lhs, detail::is_assignment(node.op.type));
    else if (leaf==RHS_NODE_TYPE && node.rhs.type != COMPOSITE_OPERATOR_TYPE)
      hash_leaf(node.rhs, false);
    else if (leaf==PARENT_NODE_TYPE)
      tools::fast_append(ptr,node.op.type);
  };

  _traverse(expression, hash_impl);

  *ptr='\0';

  return std::string(program_name);
}

}

#endif
