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
#include "isaac/exception/operation_not_supported.h"

namespace isaac
{

std::string to_string(operation_type type)
{
  switch (type)
  {
    //Function
    case ABS_TYPE : return "abs";
    case ACOS_TYPE : return "acos";
    case ASIN_TYPE : return "asin";
    case ATAN_TYPE : return "atan";
    case CEIL_TYPE : return "ceil";
    case COS_TYPE : return "cos";
    case COSH_TYPE : return "cosh";
    case EXP_TYPE : return "exp";
    case FABS_TYPE : return "fabs";
    case FLOOR_TYPE : return "floor";
    case LOG_TYPE : return "log";
    case LOG10_TYPE : return "log10";
    case SIN_TYPE : return "sin";
    case SINH_TYPE : return "sinh";
    case SQRT_TYPE : return "sqrt";
    case TAN_TYPE : return "tan";
    case TANH_TYPE : return "tanh";

    case ELEMENT_ARGFMAX_TYPE : return "argfmax";
    case ELEMENT_ARGMAX_TYPE : return "argmax";
    case ELEMENT_ARGFMIN_TYPE : return "argfmin";
    case ELEMENT_ARGMIN_TYPE : return "argmin";
    case ELEMENT_POW_TYPE : return "pow";

    //Arithmetic
    case MINUS_TYPE : return "-";
    case ASSIGN_TYPE : return "=";
    case INPLACE_ADD_TYPE : return "+=";
    case INPLACE_SUB_TYPE : return "-=";
    case ADD_TYPE : return "+";
    case SUB_TYPE : return "-";
    case MULT_TYPE : return "*";
    case ELEMENT_PROD_TYPE : return "*";
    case DIV_TYPE : return "/";
    case ELEMENT_DIV_TYPE : return "/";

    //Relational
    case NEGATE_TYPE: return "!";
    case ELEMENT_EQ_TYPE : return "==";
    case ELEMENT_NEQ_TYPE : return "!=";
    case ELEMENT_GREATER_TYPE : return ">";
    case ELEMENT_GEQ_TYPE : return ">=";
    case ELEMENT_LESS_TYPE : return "<";
    case ELEMENT_LEQ_TYPE : return "<=";

    case ELEMENT_FMAX_TYPE : return "fmax";
    case ELEMENT_FMIN_TYPE : return "fmin";
    case ELEMENT_MAX_TYPE : return "max";
    case ELEMENT_MIN_TYPE : return "min";

    default : throw operation_not_supported_exception("Unsupported operator");
  }
}


void fill(tree_node &x, invalid_node)
{
  x.type = INVALID_SUBTYPE;
  x.dtype = INVALID_NUMERIC_TYPE;
}

void fill(tree_node & x, std::size_t index)
{
  x.type = COMPOSITE_OPERATOR_TYPE;
  x.dtype = INVALID_NUMERIC_TYPE;
  x.index = index;
}

void fill(tree_node & x, placeholder index)
{
  x.type = PLACEHOLDER_TYPE;
  x.dtype = INVALID_NUMERIC_TYPE;
  x.ph = index;
}

void fill(tree_node & x, array_base const & a)
{
  x.type = DENSE_ARRAY_TYPE;
  x.dtype = a.dtype();
  x.array = (array_base*)&a;
}

void fill(tree_node & x, value_scalar const & v)
{
  x.dtype = v.dtype();
  x.type = VALUE_SCALAR_TYPE;
  x.scalar = v.values();
}

tree_node::tree_node(){}

//
op_element::op_element() {}
op_element::op_element(operation_type_family const & _type_family, operation_type const & _type) : type_family(_type_family), type(_type){}

//
expression_tree::expression_tree(placeholder const &lhs, placeholder const &rhs, const op_element &op)
 : tree_(1), root_(0), context_(NULL), dtype_(INVALID_NUMERIC_TYPE), shape_(1)
{
  fill(tree_[0].lhs, lhs);
  tree_[0].op = op;
  fill(tree_[0].rhs, rhs);
}

expression_tree::expression_tree(placeholder const &lhs, value_scalar const &rhs, const op_element &op, const numeric_type &dtype)
 : tree_(1), root_(0), context_(NULL), dtype_(dtype), shape_(1)
{
  fill(tree_[0].lhs, lhs);
  tree_[0].op = op;
  fill(tree_[0].rhs, rhs);
}

expression_tree::expression_tree(value_scalar const &lhs, placeholder const &rhs, const op_element &op, const numeric_type &dtype)
 : tree_(1), root_(0), context_(NULL), dtype_(dtype), shape_(1)
{
  fill(tree_[0].lhs, lhs);
  tree_[0].op = op;
  fill(tree_[0].rhs, rhs);
}



//expression_tree(placeholder const &lhs, placeholder const &rhs, const op_element &op);
//expression_tree(placeholder const &lhs, value_scalar const &rhs, const op_element &op, const numeric_type &dtype);

template<class LT, class RT>
expression_tree::expression_tree(LT const & lhs, RT const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, tuple const & shape) :
  tree_(1), root_(0), context_(&context), dtype_(dtype), shape_(shape)
{
  fill(tree_[0].lhs, lhs);
  tree_[0].op = op;
  tree_[0].shape = shape;
  fill(tree_[0].rhs, rhs);
}

template<class RT>
expression_tree::expression_tree(expression_tree const & lhs, RT const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, tuple const & shape) :
 tree_(lhs.tree_.size() + 1), root_(tree_.size()-1), context_(&context), dtype_(dtype), shape_(shape)
{
  std::copy(lhs.tree_.begin(), lhs.tree_.end(), tree_.begin());
  fill(tree_[root_].lhs, lhs.root_);
  tree_[root_].op = op;
  tree_[root_].shape = shape;
  fill(tree_[root_].rhs, rhs);
}

template<class LT>
expression_tree::expression_tree(LT const & lhs, expression_tree const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, tuple const & shape) :
  tree_(rhs.tree_.size() + 1), root_(tree_.size() - 1), context_(&context), dtype_(dtype), shape_(shape)
{
  std::copy(rhs.tree_.begin(), rhs.tree_.end(), tree_.begin());
  fill(tree_[root_].lhs, lhs);
  tree_[root_].op = op;
  tree_[root_].shape = shape;
  fill(tree_[root_].rhs, rhs.root_);
}

expression_tree::expression_tree(expression_tree const & lhs, expression_tree const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, tuple const & shape):
  tree_(lhs.tree_.size() + rhs.tree_.size() + 1), root_(tree_.size()-1), context_(&context), dtype_(dtype), shape_(shape)
{  
  std::size_t lsize = lhs.tree_.size();
  std::copy(lhs.tree_.begin(), lhs.tree_.end(), tree_.begin());
  std::copy(rhs.tree_.begin(), rhs.tree_.end(), tree_.begin() + lsize);
  fill(tree_[root_].lhs, lhs.root_);
  tree_[root_].op = op;
  tree_[root_].shape = shape;
  fill(tree_[root_].rhs, lsize + rhs.root_);
  for(data_type::iterator it = tree_.begin() + lsize ; it != tree_.end() - 1 ; ++it){
    if(it->lhs.type==COMPOSITE_OPERATOR_TYPE) it->lhs.index+=lsize;
    if(it->rhs.type==COMPOSITE_OPERATOR_TYPE) it->rhs.index+=lsize;
  }
  root_ = tree_.size() - 1;
}

template expression_tree::expression_tree(expression_tree const &, value_scalar const &, op_element const &,  driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(expression_tree const &, invalid_node const &, op_element const &,  driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(expression_tree const &, array_base const &,        op_element const &,  driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(expression_tree const &, placeholder const &,        op_element const &,  driver::Context const &, numeric_type const &, tuple const &);

template expression_tree::expression_tree(value_scalar const &, value_scalar const &,        op_element const &, driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(value_scalar const &, invalid_node const &,        op_element const &, driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(value_scalar const &, array_base const &,        op_element const &, driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(value_scalar const &, expression_tree const &, op_element const &,  driver::Context const &, numeric_type const &, tuple const &);

template expression_tree::expression_tree(invalid_node const &, value_scalar const &, op_element const &,  driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(invalid_node const &, expression_tree const &, op_element const &,  driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(invalid_node const &, invalid_node const &, op_element const &, driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(invalid_node const &, array_base const &,        op_element const &, driver::Context const &, numeric_type const &, tuple const &);

template expression_tree::expression_tree(array_base const &, expression_tree const &, op_element const &,         driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(array_base const &, value_scalar const &, op_element const &, driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(array_base const &, invalid_node const &, op_element const &, driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(array_base const &, array_base const &,        op_element const &, driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(array_base const &, placeholder const &, op_element const &,         driver::Context const &, numeric_type const &, tuple const &);

template expression_tree::expression_tree(placeholder const &, expression_tree const &, op_element const &,         driver::Context const &, numeric_type const &, tuple const &);
template expression_tree::expression_tree(placeholder const &, array_base const &,        op_element const &, driver::Context const &, numeric_type const &, tuple const &);

expression_tree::data_type & expression_tree::data()
{ return tree_; }

expression_tree::data_type const & expression_tree::data() const
{ return tree_; }

std::size_t expression_tree::root() const
{ return root_; }

driver::Context const & expression_tree::context() const
{ return *context_; }

numeric_type const & expression_tree::dtype() const
{ return dtype_; }

tuple expression_tree::shape() const
{ return shape_; }

int_t expression_tree::dim() const
{ return (int_t)shape_.size(); }

expression_tree expression_tree::operator-()
{ return expression_tree(*this,  invalid_node(), op_element(UNARY_TYPE_FAMILY, SUB_TYPE), *context_, dtype_, shape_); }

expression_tree expression_tree::operator!()
{ return expression_tree(*this, invalid_node(), op_element(UNARY_TYPE_FAMILY, NEGATE_TYPE), *context_, INT_TYPE, shape_); }

//

expression_tree::node const & lhs_most(expression_tree::data_type const & array, expression_tree::node const & init)
{
  expression_tree::node const * current = &init;
  while (current->lhs.type==COMPOSITE_OPERATOR_TYPE)
    current = &array[current->lhs.index];
  return *current;
}

expression_tree::node const & lhs_most(expression_tree::data_type const & array, size_t root)
{ return lhs_most(array, array[root]); }

//
expression_tree placeholder::operator=(value_scalar const & r) const { return expression_tree(*this, r, op_element(BINARY_TYPE_FAMILY,ASSIGN_TYPE), r.dtype()); }
expression_tree placeholder::operator=(expression_tree const & r) const { return expression_tree(*this, r, op_element(BINARY_TYPE_FAMILY,ASSIGN_TYPE), r.context(), r.dtype(), r.shape()); }

expression_tree placeholder::operator+=(value_scalar const & r) const { return *this = *this + r; }
expression_tree placeholder::operator-=(value_scalar const & r) const { return *this = *this - r; }
expression_tree placeholder::operator*=(value_scalar const & r) const { return *this = *this * r; }
expression_tree placeholder::operator/=(value_scalar const & r) const { return *this = *this / r; }

}
