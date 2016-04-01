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

#include <cassert>
#include <vector>
#include "isaac/array.h"
#include "isaac/value_scalar.h"
#include "isaac/symbolic/expression/expression.h"
#include "isaac/symbolic/expression/preset.h"
#include "isaac/exception/api.h"

namespace isaac
{

//
expression_tree::node::node(){}

//Constructors
expression_tree::node::node(invalid_node) : type(INVALID_SUBTYPE), dtype(INVALID_NUMERIC_TYPE)
{}

expression_tree::node::node(placeholder x) : type(PLACEHOLDER_TYPE), dtype(INVALID_NUMERIC_TYPE), shape{1}, ph(x)
{}

expression_tree::node::node(value_scalar const & x) : type(VALUE_SCALAR_TYPE), dtype(x.dtype()), shape{1}, scalar(x.values())
{}

expression_tree::node::node(array_base const & x) : type(DENSE_ARRAY_TYPE), dtype(x.dtype()), shape(x.shape())
{
  array.start = x.start();
  driver::Buffer::handle_type const & h = x.data().handle();
  switch(h.backend()){
    case driver::OPENCL: array.handle.cl = h.cl(); break;
    case driver::CUDA: array.handle.cu = h.cu(); break;
  }
  ld = x.stride();
}

expression_tree::node::node(int_t lhs, op_element op, int_t rhs, numeric_type dt, tuple const & sh)
{
  type = COMPOSITE_OPERATOR_TYPE;
  dtype = dt;
  shape = sh;
  binary_operator.lhs = lhs;
  binary_operator.op = op;
  binary_operator.rhs = rhs;
}

//
expression_tree::expression_tree(node const & lhs, node const & rhs, op_element const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape) :
  root_(2), context_(context)
{
  tree_.reserve(3);
  tree_.push_back(std::move(lhs));
  tree_.push_back(std::move(rhs));
  tree_.emplace_back(node(0, op, 1, dtype, shape));
}

expression_tree::expression_tree(expression_tree const & lhs, node const & rhs, op_element const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape) :
 tree_(lhs.tree_.size() + 2), root_(tree_.size() - 1), context_(context)
{
  std::move(lhs.tree_.begin(), lhs.tree_.end(), tree_.begin());
  tree_[root_ - 1] = rhs;
  tree_[root_] = node(lhs.root_, op, root_ - 1, dtype, shape);
}

expression_tree::expression_tree(node const & lhs, expression_tree const & rhs, op_element const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape) :
  tree_(rhs.tree_.size() + 2), root_(tree_.size() - 1), context_(context)
{
  std::move(rhs.tree_.begin(), rhs.tree_.end(), tree_.begin());
  tree_[root_ - 1] = lhs;
  tree_[root_] = node(root_ - 1, op, rhs.root_, dtype, shape);
}

expression_tree::expression_tree(expression_tree const & lhs, expression_tree const & rhs, op_element const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape):
  tree_(lhs.tree_.size() + rhs.tree_.size() + 1), root_(tree_.size()-1), context_(context)
{  
  std::size_t lsize = lhs.tree_.size();
  std::move(lhs.tree_.begin(), lhs.tree_.end(), tree_.begin());
  std::move(rhs.tree_.begin(), rhs.tree_.end(), tree_.begin() + lsize);
  tree_[root_] = node(lhs.root_, op, lsize + rhs.root_, dtype, shape);
  for(data_type::iterator it = tree_.begin() + lsize ; it != tree_.end() - 1 ; ++it){
    if(it->type==COMPOSITE_OPERATOR_TYPE){
      it->binary_operator.lhs += lsize;
      it->binary_operator.rhs += lsize;
    }
  }
}

expression_tree::data_type const & expression_tree::data() const
{ return tree_; }

std::size_t expression_tree::root() const
{ return root_; }

driver::Context const & expression_tree::context() const
{ return *context_; }

numeric_type const & expression_tree::dtype() const
{ return tree_[root_].dtype; }

tuple expression_tree::shape() const
{ return tree_[root_].shape; }

int_t expression_tree::dim() const
{ return (int_t)shape().size(); }

expression_tree expression_tree::operator-()
{ return expression_tree(*this,  invalid_node(), op_element(UNARY_ARITHMETIC, SUB_TYPE), context_, dtype(), shape()); }

expression_tree expression_tree::operator!()
{ return expression_tree(*this, invalid_node(), op_element(UNARY_ARITHMETIC, NEGATE_TYPE), context_, INT_TYPE, shape()); }

expression_tree::node const & expression_tree::operator[](size_t idx) const
{ return tree_[idx]; }

expression_tree::node & expression_tree::operator[](size_t idx)
{ return tree_[idx]; }

//
expression_tree placeholder::operator=(value_scalar const & r) const { return expression_tree(*this, r, op_element(BINARY_ARITHMETIC,ASSIGN_TYPE), NULL, r.dtype(), {1}); }
expression_tree placeholder::operator=(expression_tree const & r) const { return expression_tree(*this, r, op_element(BINARY_ARITHMETIC,ASSIGN_TYPE), &r.context(), r.dtype(), r.shape()); }

expression_tree placeholder::operator+=(value_scalar const & r) const { return *this = *this + r; }
expression_tree placeholder::operator-=(value_scalar const & r) const { return *this = *this - r; }
expression_tree placeholder::operator*=(value_scalar const & r) const { return *this = *this * r; }
expression_tree placeholder::operator/=(value_scalar const & r) const { return *this = *this / r; }

}
