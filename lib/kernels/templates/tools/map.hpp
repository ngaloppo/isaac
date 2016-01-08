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

#include <string>
#include <vector>
#include "isaac/kernels/symbolic_object.h"
#include "isaac/kernels/parse.h"

namespace isaac
{

namespace templates
{

class map_functor : public traversal_functor
{

  std::shared_ptr<symbolic::object> create(numeric_type dtype, values_holder) const
  {
    std::string strdtype = to_string(dtype);
    return std::shared_ptr<symbolic::object>(new symbolic::host_scalar(strdtype, binder_.get()));
  }

  std::shared_ptr<symbolic::object> create(array_base const * a, bool is_assigned)  const
  {
    std::string dtype = to_string(a->dtype());
    unsigned int id = binder_.get(a, is_assigned);
    return std::shared_ptr<symbolic::object>(new symbolic::buffer(dtype, id, a->shape()));
  }

  std::shared_ptr<symbolic::object> create(tree_node const & lhs_rhs, bool is_assigned = false) const
  {
    switch(lhs_rhs.subtype)
    {
      case VALUE_SCALAR_TYPE: return create(lhs_rhs.dtype, lhs_rhs.scalar);
      case DENSE_ARRAY_TYPE: return create(lhs_rhs.array, is_assigned);
      case PLACEHOLDER_TYPE: return std::shared_ptr<symbolic::object>(new symbolic::placeholder(lhs_rhs.ph.level));
      default: throw "";
    }
  }

  template<class T, class... Args>
  typename symbolic::mapping_type::value_type make_pair(symbolic::mapping_type::key_type const & key, Args&&... args) const
  {
    return {key, std::shared_ptr<symbolic::object>(new T(std::forward<Args>(args)...))};
  }

public:
  map_functor(symbolic_binder & binder, symbolic::mapping_type & mapping, const driver::Device &device)
      : binder_(binder), mapping_(mapping), device_(device)
  {
  }

  void operator()(isaac::expression_tree const & expression, size_t idx, leaf_t leaf) const
  {
    symbolic::mapping_type::key_type key(idx, leaf);
    expression_tree::node const & node = expression.tree()[idx];
    std::string dtype = to_string(expression.dtype());

    if (leaf == LHS_NODE_TYPE && node.lhs.subtype != COMPOSITE_OPERATOR_TYPE)
      mapping_.insert(symbolic::mapping_type::value_type(key, create(node.lhs, detail::is_assignment(node.op))));
    else if (leaf == RHS_NODE_TYPE && node.rhs.subtype != COMPOSITE_OPERATOR_TYPE)
      mapping_.insert(symbolic::mapping_type::value_type(key, create(node.rhs)));
    else if (leaf== PARENT_NODE_TYPE)
    {
      unsigned int id = binder_.get();
      //1D Reduction
      if (node.op.type_family==VECTOR_DOT_TYPE_FAMILY)
        mapping_.insert(make_pair<symbolic::reduce_1d>(key, dtype, id, idx, node.op));
      //2D reduction
      else if (node.op.type_family==ROWS_DOT_TYPE_FAMILY || node.op.type_family==COLUMNS_DOT_TYPE_FAMILY)
        mapping_.insert(make_pair<symbolic::reduce_2d>(key, dtype, id, idx, node.op));
      //Index modifier
      else if (node.op.type==VDIAG_TYPE)
        mapping_.insert(make_pair<symbolic::diag_matrix>(key, dtype, id, idx, mapping_));
      else if (node.op.type==MATRIX_DIAG_TYPE)
        mapping_.insert(make_pair<symbolic::diag_vector>(key, dtype, id, idx, mapping_));
      else if (node.op.type==MATRIX_ROW_TYPE)
        mapping_.insert(make_pair<symbolic::matrix_row>(key, dtype, id, idx, mapping_));
      else if (node.op.type==MATRIX_COLUMN_TYPE)
        mapping_.insert(make_pair<symbolic::matrix_column>(key, dtype, id, idx, mapping_));
      else if(node.op.type==ACCESS_INDEX_TYPE)
        mapping_.insert(make_pair<symbolic::array_access>(key, dtype, id, idx, mapping_));
      else if(node.op.type==RESHAPE_TYPE)
        mapping_.insert(make_pair<symbolic::reshape>(key, dtype, id, idx, expression, mapping_));
      else if (detail::is_cast(node.op))
        mapping_.insert(make_pair<symbolic::cast>(key, node.op.type, id));
    }
  }
private:
  symbolic_binder & binder_;
  symbolic::mapping_type & mapping_;
  driver::Device const & device_;
};


}

}
