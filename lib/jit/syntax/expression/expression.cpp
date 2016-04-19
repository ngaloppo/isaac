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
#include "isaac/scalar.h"
#include "isaac/jit/syntax/expression/expression.h"
#include "isaac/jit/syntax/expression/preset.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{

//Tokens
token::token() {}

token::token(token_family const & _family, token_type const & _type) : family(_family), type(_type){}

std::string to_string(token_type type)
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

    //dot
    case MATRIX_PRODUCT_NN_TYPE: return "matmatNN";
    case MATRIX_PRODUCT_NT_TYPE: return "matmatNT";
    case MATRIX_PRODUCT_TN_TYPE: return "matmatTN";
    case MATRIX_PRODUCT_TT_TYPE: return "matmatTT";

    //others
    case RESHAPE_TYPE: return "reshape";
    case TRANS_TYPE: return "trans";

    default : throw "Unsupported operator";
  }
}

bool is_assignment(token_type op)
{
  return op== ASSIGN_TYPE
      || op== INPLACE_ADD_TYPE
      || op== INPLACE_SUB_TYPE;
}

bool is_operator(token_type op)
{
  return is_assignment(op)
      || op == ADD_TYPE
      || op == SUB_TYPE
      || op == ELEMENT_PROD_TYPE
      || op == ELEMENT_DIV_TYPE
      || op == MULT_TYPE
      || op == DIV_TYPE
      || op == ELEMENT_EQ_TYPE
      || op == ELEMENT_NEQ_TYPE
      || op == ELEMENT_GREATER_TYPE
      || op == ELEMENT_LESS_TYPE
      || op == ELEMENT_GEQ_TYPE
      || op == ELEMENT_LEQ_TYPE ;
}

bool is_cast(token_type op)
{
  return op == CAST_BOOL_TYPE
      || op == CAST_CHAR_TYPE
      || op == CAST_UCHAR_TYPE
      || op == CAST_SHORT_TYPE
      || op == CAST_USHORT_TYPE
      || op == CAST_INT_TYPE
      || op == CAST_UINT_TYPE
      || op == CAST_LONG_TYPE
      || op == CAST_ULONG_TYPE
      || op == CAST_FLOAT_TYPE
      || op == CAST_DOUBLE_TYPE
      ;
}

bool is_function(token_type op)
{
  return is_cast(op)
      || op == ABS_TYPE
      || op == ACOS_TYPE
      || op == ASIN_TYPE
      || op == ATAN_TYPE
      || op == CEIL_TYPE
      || op == COS_TYPE
      || op == COSH_TYPE
      || op == EXP_TYPE
      || op == FABS_TYPE
      || op == FLOOR_TYPE
      || op == LOG_TYPE
      || op == LOG10_TYPE
      || op == SIN_TYPE
      || op == SINH_TYPE
      || op == SQRT_TYPE
      || op == TAN_TYPE
      || op == TANH_TYPE

      || op == ELEMENT_POW_TYPE
      || op == ELEMENT_FMAX_TYPE
      || op == ELEMENT_FMIN_TYPE
      || op == ELEMENT_MAX_TYPE
      || op == ELEMENT_MIN_TYPE;

}

bool is_indexing(token_type op)
{
  return op == ELEMENT_ARGFMAX_TYPE
      || op == ELEMENT_ARGMAX_TYPE
      || op == ELEMENT_ARGFMIN_TYPE
      || op == ELEMENT_ARGMIN_TYPE;
}


//
expression_tree::node::node(){}

//Constructors
expression_tree::node::node(invalid_node) : type(INVALID_SUBTYPE), dtype(INVALID_NUMERIC_TYPE)
{}

expression_tree::node::node(scalar const & x) : type(VALUE_SCALAR_TYPE), dtype(x.dtype()), shape{1}, value(x.values())
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

expression_tree::node::node(int_t lhs, token op, int_t rhs, numeric_type dt, tuple const & sh)
{
  type = COMPOSITE_OPERATOR_TYPE;
  dtype = dt;
  shape = sh;
  binary_operator.lhs = lhs;
  binary_operator.op = op;
  binary_operator.rhs = rhs;
}

//
expression_tree::expression_tree(node const & lhs, node const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape) :
  root_(2), context_(context)
{
  tree_.reserve(3);
  tree_.push_back(std::move(lhs));
  tree_.push_back(std::move(rhs));
  tree_.emplace_back(node(0, op, 1, dtype, shape));
}

expression_tree::expression_tree(expression_tree const & lhs, node const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape) :
 tree_(lhs.tree_.size() + 2), root_(tree_.size() - 1), context_(context)
{
  std::move(lhs.tree_.begin(), lhs.tree_.end(), tree_.begin());
  tree_[root_ - 1] = rhs;
  tree_[root_] = node(lhs.root_, op, root_ - 1, dtype, shape);
}

expression_tree::expression_tree(node const & lhs, expression_tree const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape) :
  tree_(rhs.tree_.size() + 2), root_(tree_.size() - 1), context_(context)
{
  std::move(rhs.tree_.begin(), rhs.tree_.end(), tree_.begin());
  tree_[root_ - 1] = lhs;
  tree_[root_] = node(root_ - 1, op, rhs.root_, dtype, shape);
}

expression_tree::expression_tree(expression_tree const & lhs, expression_tree const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape):
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
{ return expression_tree(*this,  invalid_node(), token(UNARY_ARITHMETIC, SUB_TYPE), context_, dtype(), shape()); }

expression_tree expression_tree::operator!()
{ return expression_tree(*this, invalid_node(), token(UNARY_ARITHMETIC, NEGATE_TYPE), context_, INT_TYPE, shape()); }

expression_tree::node const & expression_tree::operator[](size_t idx) const
{ return tree_[idx]; }

expression_tree::node & expression_tree::operator[](size_t idx)
{ return tree_[idx]; }

//io
#define ISAAC_MAP_TO_STRING(NAME) case NAME: return #NAME

inline std::string to_string(const token& op)
{
  std::string res = to_string(op.type);
  if(op.family==REDUCE) res = "reduce<" + res + ">";
  if(op.family==REDUCE_ROWS) res = "reduce<" + res + ", rows>";
  if(op.family==REDUCE_COLUMNS) res = "reduce<" + res + ", cols>";
  return res;
}

inline std::string to_string(const expression_tree::node &node)
{
  if(node.type==COMPOSITE_OPERATOR_TYPE)
  {
    std::string lhs = tools::to_string(node.binary_operator.lhs);
    std::string op = to_string(node.binary_operator.op);
    std::string rhs = tools::to_string(node.binary_operator.rhs);
    return"node (" + lhs + ", " + op + ", " + rhs + ")";
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

namespace detail
{
  /** @brief Recursive worker routine for printing a whole expression_tree */
  inline void print_node(std::ostream & os, isaac::expression_tree const & s, size_t index, size_t indent = 0)
  {
    expression_tree::data_type const & data = s.data();
    expression_tree::node const & node = data[index];

    for (size_t i=0; i<indent; ++i)
      os << " ";

    os << "Node " << index << ": " << to_string(node) << std::endl;

    if (node.type == COMPOSITE_OPERATOR_TYPE)
    {
      print_node(os, s, node.binary_operator.lhs, indent+1);
      print_node(os, s, node.binary_operator.rhs, indent+1);
    }
  }
}

std::string to_string(isaac::expression_tree const & s)
{
  std::ostringstream os;
  detail::print_node(os, s, s.root());
  return os.str();
}

}
