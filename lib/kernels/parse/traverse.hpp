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

#ifndef ISAAC_KERNELS_PARSE_TRAVERSE_HPP
#define ISAAC_KERNELS_PARSE_TRAVERSE_HPP

#include <cstring>

#include "isaac/kernels/parse.h"

namespace isaac
{

namespace detail
{
  template<class FUN>
  inline void _traverse(expression_tree const & tree, size_t root, FUN const & fun)
  {
    expression_tree::node const & root_node = tree.tree()[root];

    //Lhs:
    if (root_node.lhs.subtype==COMPOSITE_OPERATOR_TYPE)
      _traverse(tree, root_node.lhs.index, fun);
    if (root_node.lhs.subtype != INVALID_SUBTYPE)
      fun(root, LHS_NODE_TYPE);

    //Rhs:
    if (root_node.rhs.subtype!=INVALID_SUBTYPE)
    {
      if (root_node.rhs.subtype==COMPOSITE_OPERATOR_TYPE)
        _traverse(tree, root_node.rhs.index, fun);
      if (root_node.rhs.subtype != INVALID_SUBTYPE)
        fun(root, RHS_NODE_TYPE);
    }

    fun(root, PARENT_NODE_TYPE);
  }
}

template<class FUN>
inline void _traverse(expression_tree const & expression, size_t idx, leaf_t leaf, FUN const & fun)
{
  expression_tree::node const & root = expression.tree()[idx];
  if(leaf==RHS_NODE_TYPE && root.rhs.subtype==COMPOSITE_OPERATOR_TYPE)
    detail::_traverse(expression, root.rhs.index, fun);
  else if(leaf==LHS_NODE_TYPE && root.lhs.subtype==COMPOSITE_OPERATOR_TYPE)
    detail::_traverse(expression, root.lhs.index, fun);
  else if(leaf==PARENT_NODE_TYPE)
    detail::_traverse(expression, idx, fun);
  else
    fun(idx, leaf);
}

template<class FUN>
inline void _traverse(expression_tree const & tree, FUN const & fun)
{
  return _traverse(tree, tree.root(), PARENT_NODE_TYPE, fun);
}

}

#endif
