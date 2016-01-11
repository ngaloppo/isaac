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

#ifndef ISAAC_KERNELS_PARSE_SYMBOLIZE_HPP
#define ISAAC_KERNELS_PARSE_SYMBOLIZE_HPP

#include <cstring>

#include "isaac/kernels/symbolic_object.h"
#include "isaac/kernels/parse.h"
#include "traverse.hpp"

namespace isaac
{

template<class T, class... Args>
std::shared_ptr<symbolic::object> make_symbolic(Args&&... args)
{ return std::shared_ptr<symbolic::object>(new T(std::forward<Args>(args)...)); }

inline symbolic::mapping_type symbolize(binding_policy_t binding_policy, isaac::expression_tree const & expression)
{
  //binder
  symbolic::mapping_type result;
  std::unique_ptr<symbolic_binder> binder;
  if (binding_policy==BIND_SEQUENTIAL)
      binder.reset(new bind_sequential());
  else
      binder.reset(new bind_independent());

  //make_leaf
  auto symbolize_leaf = [&](std::string const & dtype, tree_node const & leaf, bool is_assigned)
  {
    if(leaf.subtype==VALUE_SCALAR_TYPE)
      return make_symbolic<symbolic::host_scalar>(dtype, binder->get());
    else if(leaf.subtype==DENSE_ARRAY_TYPE)
      return make_symbolic<symbolic::buffer>(dtype, binder->get(leaf.array, is_assigned), leaf.array->shape());
    else if(leaf.subtype==PLACEHOLDER_TYPE)
      return make_symbolic<symbolic::placeholder>(leaf.ph.level);
    else
      throw;
  };

  //make_impl
  auto symbolize_impl = [&](size_t idx, leaf_t leaf)
  {
    symbolic::mapping_type::key_type key(idx, leaf);
    expression_tree::node const & node = expression.tree()[idx];
    std::string dtype = to_string(expression.dtype());

    if (leaf == LHS_NODE_TYPE && node.lhs.subtype != COMPOSITE_OPERATOR_TYPE)
      result.insert({key, symbolize_leaf(dtype, node.lhs, detail::is_assignment(node.op.type))});
    else if (leaf == RHS_NODE_TYPE && node.rhs.subtype != COMPOSITE_OPERATOR_TYPE)
      result.insert({key, symbolize_leaf(dtype, node.rhs, false)});
    else if (leaf== PARENT_NODE_TYPE)
    {
      unsigned int id = binder->get();
      //Unary arithmetic
      if(node.op.type_family==UNARY_TYPE_FAMILY)
        result.insert({key, make_symbolic<symbolic::unary_node>(node.op.type, idx, expression, result)});
      //Binary arithmetic
      if(node.op.type_family==BINARY_TYPE_FAMILY)
        result.insert({key, make_symbolic<symbolic::binary_node>(node.op.type, idx, expression, result)});
      //1D Reduction
      if (node.op.type_family==VECTOR_DOT_TYPE_FAMILY)
        result.insert({key, make_symbolic<symbolic::reduce_1d>(dtype, id, idx, node.op)});
      //2D reduction
      else if (node.op.type_family==ROWS_DOT_TYPE_FAMILY || node.op.type_family==COLUMNS_DOT_TYPE_FAMILY)
        result.insert({key, make_symbolic<symbolic::reduce_2d>(dtype, id, idx, node.op)});
      //Index modifier
      else if (node.op.type==VDIAG_TYPE)
        result.insert({key, make_symbolic<symbolic::diag_matrix>(dtype, id, idx, result)});
      else if (node.op.type==MATRIX_DIAG_TYPE)
        result.insert({key, make_symbolic<symbolic::diag_vector>(dtype, id, idx, result)});
      else if (node.op.type==MATRIX_ROW_TYPE)
        result.insert({key, make_symbolic<symbolic::matrix_row>(dtype, id, idx, result)});
      else if (node.op.type==MATRIX_COLUMN_TYPE)
        result.insert({key, make_symbolic<symbolic::matrix_column>(dtype, id, idx, result)});
      else if(node.op.type==ACCESS_INDEX_TYPE)
        result.insert({key, make_symbolic<symbolic::array_access>(dtype, id, idx, result)});
      else if(node.op.type==RESHAPE_TYPE)
        result.insert({key, make_symbolic<symbolic::reshape>(dtype, id, idx, expression, result)});
      else if (detail::is_cast(node.op))
        result.insert({key, make_symbolic<symbolic::cast>(node.op.type, id)});
    }
  };

  //traverse
  _traverse(expression, symbolize_impl);

  return result;
}

}

#endif
