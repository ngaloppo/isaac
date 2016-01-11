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

namespace detail
{

class symbolize
{

  template<class T, class... Args>
  std::shared_ptr<symbolic::object> create(Args&&... args) const
  {
    return std::shared_ptr<symbolic::object>(new T(std::forward<Args>(args)...));
  }

  std::shared_ptr<symbolic::object> create(std::string const & dtype, tree_node const & lhs_rhs, bool is_assigned = false) const
  {
    if(lhs_rhs.subtype==VALUE_SCALAR_TYPE) return create<symbolic::host_scalar>(dtype, binder_.get());
    else if(lhs_rhs.subtype==DENSE_ARRAY_TYPE) return create<symbolic::buffer>(dtype, binder_.get(lhs_rhs.array, is_assigned), lhs_rhs.array->shape());
    else if(lhs_rhs.subtype==PLACEHOLDER_TYPE) return create<symbolic::placeholder>(lhs_rhs.ph.level);
    else throw;
  }

public:
  symbolize(symbolic_binder & binder, isaac::expression_tree const & expression, symbolic::mapping_type & mapping, const driver::Device &device)
      : binder_(binder), expression_(expression), mapping_(mapping), device_(device)
  {
  }

  void operator()(size_t idx, leaf_t leaf) const
  {
    symbolic::mapping_type::key_type key(idx, leaf);
    expression_tree::node const & node = expression_.tree()[idx];
    std::string dtype = to_string(expression_.dtype());

    if (leaf == LHS_NODE_TYPE && node.lhs.subtype != COMPOSITE_OPERATOR_TYPE)
      mapping_.insert({key, create(dtype, node.lhs, detail::is_assignment(node.op.type))});
    else if (leaf == RHS_NODE_TYPE && node.rhs.subtype != COMPOSITE_OPERATOR_TYPE)
      mapping_.insert({key, create(dtype, node.rhs)});
    else if (leaf== PARENT_NODE_TYPE)
    {
      unsigned int id = binder_.get();
      //Unary arithmetic
      if(node.op.type_family==UNARY_TYPE_FAMILY)
        mapping_.insert({key, create<symbolic::unary_node>(node.op.type, idx, expression_, mapping_)});
      //Binary arithmetic
      if(node.op.type_family==BINARY_TYPE_FAMILY)
        mapping_.insert({key, create<symbolic::binary_node>(node.op.type, idx, expression_, mapping_)});
      //1D Reduction
      if (node.op.type_family==VECTOR_DOT_TYPE_FAMILY)
        mapping_.insert({key, create<symbolic::reduce_1d>(dtype, id, idx, node.op)});
      //2D reduction
      else if (node.op.type_family==ROWS_DOT_TYPE_FAMILY || node.op.type_family==COLUMNS_DOT_TYPE_FAMILY)
        mapping_.insert({key, create<symbolic::reduce_2d>(dtype, id, idx, node.op)});
      //Index modifier
      else if (node.op.type==VDIAG_TYPE)
        mapping_.insert({key, create<symbolic::diag_matrix>(dtype, id, idx, mapping_)});
      else if (node.op.type==MATRIX_DIAG_TYPE)
        mapping_.insert({key, create<symbolic::diag_vector>(dtype, id, idx, mapping_)});
      else if (node.op.type==MATRIX_ROW_TYPE)
        mapping_.insert({key, create<symbolic::matrix_row>(dtype, id, idx, mapping_)});
      else if (node.op.type==MATRIX_COLUMN_TYPE)
        mapping_.insert({key, create<symbolic::matrix_column>(dtype, id, idx, mapping_)});
      else if(node.op.type==ACCESS_INDEX_TYPE)
        mapping_.insert({key, create<symbolic::array_access>(dtype, id, idx, mapping_)});
      else if(node.op.type==RESHAPE_TYPE)
        mapping_.insert({key, create<symbolic::reshape>(dtype, id, idx, expression_, mapping_)});
      else if (detail::is_cast(node.op))
        mapping_.insert({key, create<symbolic::cast>(node.op.type, id)});
    }
  }
private:
  symbolic_binder& binder_;
  isaac::expression_tree const & expression_;
  symbolic::mapping_type & mapping_;
  driver::Device const & device_;
};

}

inline symbolic::mapping_type symbolize(binding_policy_t binding_policy, isaac::expression_tree const & expression, driver::Device const & device)
{
  symbolic::mapping_type result;
  std::unique_ptr<symbolic_binder> binder;
  if (binding_policy==BIND_SEQUENTIAL)
      binder.reset(new bind_sequential());
  else
      binder.reset(new bind_independent());
  _traverse(expression, detail::symbolize(*binder, expression, result, device));
  return result;
}

}

#endif
