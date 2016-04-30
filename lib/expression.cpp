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
#include <map>
#include "isaac/array.h"
#include "isaac/scalar.h"
#include "isaac/expression.h"
#include "isaac/jit/syntax/expression/preset.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{

//
expression::node::node(){}

//Constructors
expression::node::node(invalid_node) : type(INVALID_SUBTYPE), dtype(INVALID_NUMERIC_TYPE)
{}

expression::node::node(scalar const & x) : type(VALUE_SCALAR_TYPE), dtype(x.dtype()), shape{1}, value(x.values())
{}

expression::node::node(array_base const & x) : type(DENSE_ARRAY_TYPE), dtype(x.dtype()), shape(x.shape())
{
  array.start = x.start();
  driver::Buffer::handle_type const & h = x.data().handle();
  switch(h.backend()){
    case driver::OPENCL: array.handle.cl = h.cl(); break;
    case driver::CUDA: array.handle.cu = h.cu(); break;
  }
  ld = x.stride();
}

expression::node::node(int_t lhs, token op, int_t rhs, numeric_type dt, tuple const & sh)
{
  type = COMPOSITE_OPERATOR_TYPE;
  dtype = dt;
  shape = sh;
  binary_operator.lhs = lhs;
  binary_operator.op = op;
  binary_operator.rhs = rhs;
}

//
expression::expression(node const & lhs, node const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape) :
  root_(2), context_(context), init_(false)
{
  tree_.reserve(3);
  tree_.push_back(std::move(lhs));
  tree_.push_back(std::move(rhs));
  tree_.emplace_back(node(0, op, 1, dtype, shape));
}

expression::expression(expression const & lhs, node const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape) :
 tree_(lhs.tree_.size() + 2), root_(tree_.size() - 1), context_(context), init_(false)
{
  std::move(lhs.tree_.begin(), lhs.tree_.end(), tree_.begin());
  tree_[root_ - 1] = rhs;
  tree_[root_] = node(lhs.root_, op, root_ - 1, dtype, shape);
}

expression::expression(node const & lhs, expression const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape) :
  tree_(rhs.tree_.size() + 2), root_(tree_.size() - 1), context_(context), init_(false)
{
  std::move(rhs.tree_.begin(), rhs.tree_.end(), tree_.begin());
  tree_[root_ - 1] = lhs;
  tree_[root_] = node(root_ - 1, op, rhs.root_, dtype, shape);
}

expression::expression(expression const & lhs, expression const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape):
  tree_(lhs.tree_.size() + rhs.tree_.size() + 1), root_(tree_.size()-1), context_(context), init_(false)
{  
  size_t lsize = lhs.tree_.size();
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

void expression::init()
{
  //compare memory handles
  auto cmp = [&](handle_t const & x, handle_t const & y){
    if(context_->backend()==driver::OPENCL)
      return x.cl < y.cl;
    else
      return x.cu < y.cu;
  };
  std::map<handle_t, size_t, decltype(cmp)> handles(cmp);
  size_t current = 0;
  //traversal functor
  auto fun = [&](size_t root, size_t)
  {
    expression::node& node = tree_[root];
    if(node.type==DENSE_ARRAY_TYPE){
      auto it = handles.insert({node.array.handle, current++}).first;
      node.id = it->second;
    }
    else
      node.id = current++;
  };
  traverse_dfs(*this, fun);
}

size_t expression::size() const
{ return tree_.size(); }

size_t expression::root() const
{ return root_; }

driver::Context const & expression::context() const
{ return *context_; }

numeric_type const & expression::dtype() const
{ return tree_[root_].dtype; }

tuple expression::shape() const
{ return tree_[root_].shape; }

int_t expression::dim() const
{ return (int_t)shape().size(); }

expression expression::operator-()
{ return expression(*this,  invalid_node(), {UNARY_ARITHMETIC, SUB_TYPE}, context_, dtype(), shape()); }

expression expression::operator!()
{ return expression(*this, invalid_node(), {UNARY_ARITHMETIC, NEGATE_TYPE}, context_, INT_TYPE, shape()); }

expression::node const & expression::operator[](size_t idx) const
{ return tree_[idx]; }

expression::node & expression::operator[](size_t idx)
{ return tree_[idx]; }

//Evaluate
std::string eval(token_type type)
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
    case ASSIGN_TYPE : return "=";
    case ADD_TYPE : return "+";
    case SUB_TYPE : return "-";
    case MULT_TYPE : return "*";
    case DIV_TYPE : return "/";

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

    case CAST_BOOL_TYPE : return "(bool)";
    case CAST_CHAR_TYPE : return "(char)";
    case CAST_UCHAR_TYPE : return "(uchar)";
    case CAST_SHORT_TYPE : return "(short)";
    case CAST_USHORT_TYPE : return "(ushort)";
    case CAST_INT_TYPE: return "(int)";
    case CAST_UINT_TYPE : return "(uint)";
    case CAST_LONG_TYPE : return "(long)";
    case CAST_ULONG_TYPE : return "(ulong)";
    case CAST_FLOAT_TYPE : return "(float)";
    case CAST_DOUBLE_TYPE : return "(double)";

    default : throw "Not compound operator";
  }
}

//predicates
bool is_operator(token_type op)
{
  return op == ASSIGN_TYPE
      || op == ADD_TYPE
      || op == SUB_TYPE
      || op == MULT_TYPE
      || op == DIV_TYPE
      || op == ELEMENT_EQ_TYPE
      || op == ELEMENT_NEQ_TYPE
      || op == ELEMENT_GREATER_TYPE
      || op == ELEMENT_LESS_TYPE
      || op == ELEMENT_GEQ_TYPE
      || op == ELEMENT_LEQ_TYPE ;
}

bool is_indexing(token_type op)
{
  return op == ELEMENT_ARGFMAX_TYPE
      || op == ELEMENT_ARGMAX_TYPE
      || op == ELEMENT_ARGFMIN_TYPE
      || op == ELEMENT_ARGMIN_TYPE;
}


//to_string
std::string to_string(token_type type)
{
  switch (type)
  {
    //Unary functions
    case ABS_TYPE : return "ABS";
    case ACOS_TYPE : return "ACOS";
    case ASIN_TYPE : return "ASIN";
    case ATAN_TYPE : return "ATAN";
    case CEIL_TYPE : return "CEIL";
    case COS_TYPE : return "COS";
    case COSH_TYPE : return "COSH";
    case EXP_TYPE : return "EXP";
    case FABS_TYPE : return "FABS";
    case FLOOR_TYPE : return "FLOOR";
    case LOG_TYPE : return "LOG";
    case LOG10_TYPE : return "LOG10";
    case SIN_TYPE : return "SIN";
    case SINH_TYPE : return "SINH";
    case SQRT_TYPE : return "SQRT";
    case TAN_TYPE : return "TAN";
    case TANH_TYPE : return "TANH";
    //Binary functions
    case ELEMENT_POW_TYPE : return "POW";
    case ELEMENT_FMAX_TYPE : return "FMAX";
    case ELEMENT_FMIN_TYPE : return "FMIN";
    case ELEMENT_MAX_TYPE : return "MAX";
    case ELEMENT_MIN_TYPE : return "MIN";
    //Reduction operators
    case ELEMENT_ARGFMAX_TYPE : return "ARGFMAX";
    case ELEMENT_ARGMAX_TYPE : return "ARGMAX";
    case ELEMENT_ARGFMIN_TYPE : return "ARGFMIN";
    case ELEMENT_ARGMIN_TYPE : return "ARGMIN";
    //Cast
    case CAST_BOOL_TYPE : return "TO_BOOL";
    case CAST_CHAR_TYPE : return "TO_CHAR";
    case CAST_UCHAR_TYPE : return "TO_UCHAR";
    case CAST_SHORT_TYPE : return "TO_SHORT";
    case CAST_USHORT_TYPE : return "TO_USHORT";
    case CAST_INT_TYPE: return "TO_INT";
    case CAST_UINT_TYPE : return "TO_UINT";
    case CAST_LONG_TYPE : return "TO_LONG";
    case CAST_ULONG_TYPE : return "TO_ULONG";
    case CAST_FLOAT_TYPE : return "TO_FLOAT";
    case CAST_DOUBLE_TYPE : return "TO_DOUBLE";
    //Special
    case MATRIX_PRODUCT_NN_TYPE: return "DOT_NN";
    case MATRIX_PRODUCT_NT_TYPE: return "DOT_NT";
    case MATRIX_PRODUCT_TN_TYPE: return "DOT_TN";
    case MATRIX_PRODUCT_TT_TYPE: return "DOT_TT";
    case RESHAPE_TYPE: return "RESHAPE";
    case TRANS_TYPE: return "TRANS";
    //Arithmetic
    case ASSIGN_TYPE : return "SET";
    case ADD_TYPE : return "ADD";
    case SUB_TYPE : return "SUB";
    case MULT_TYPE : return "MULT";
    case DIV_TYPE : return "DIV";
    //Relational
    case NEGATE_TYPE: return "NOT";
    case ELEMENT_EQ_TYPE : return "EQ";
    case ELEMENT_NEQ_TYPE : return "NEQ";
    case ELEMENT_GREATER_TYPE : return "GT";
    case ELEMENT_GEQ_TYPE : return "GEQ";
    case ELEMENT_LESS_TYPE : return "LT";
    case ELEMENT_LEQ_TYPE : return "LEQ";
    //Default
    default : throw "Unsupported operator";
  }
}

std::string to_string(const token& op)
{
  std::string res = to_string(op.type);
  if(op.family==REDUCE) res = "RED<" + res + ">";
  if(op.family==REDUCE_ROWS) res = "RED<" + res + ", 1>";
  if(op.family==REDUCE_COLUMNS) res = "RED<" + res + ", 0>";
  return res;
}

std::string to_string(const expression::node &node)
{
  if(node.type==COMPOSITE_OPERATOR_TYPE)
  {
    std::string lhs = tools::to_string(node.binary_operator.lhs);
    std::string op = to_string(node.binary_operator.op);
    std::string rhs = tools::to_string(node.binary_operator.rhs);
    return"compound(" + lhs + ", " + op + ", " + rhs + ")";
  }
  switch(node.type)
  {
    case INVALID_SUBTYPE:
      return "empty";
    case VALUE_SCALAR_TYPE:
      return "scalar";
    case DENSE_ARRAY_TYPE:
      return "array";
    default:
      return "unknown";
  }
}

std::string to_string(isaac::expression const & tree)
{
  std::ostringstream os;
  auto fun = [&](size_t index, size_t depth){
    for (size_t i=0; i<depth; ++i)
      os << " ";
    os << "Node " << index << ": " << to_string(tree[index]) << std::endl;
  };
  traverse_bfs(tree, fun);
  return os.str();
}

}
