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

#ifndef _ISAAC_SYMBOLIC_EXPRESSION_H
#define _ISAAC_SYMBOLIC_EXPRESSION_H

#include <utility>
#include <vector>
#include <list>
#include <memory>
#include <iostream>

#include "isaac/driver/external/CUDA/cuda.h"
#include "isaac/driver/external/CL/cl.h"
#include "isaac/tools/cpp/tuple.hpp"
#include "isaac/scalar.h"


namespace isaac
{

namespace driver
{
  class Context;
}

class array_base;

struct invalid_node{};

/** @brief Optimization enum for grouping operations into unary or binary operations. Just for optimization of lookups. */
enum token_family
{
  INVALID_ = 0,

  // BLAS1-type
  UNARY_ARITHMETIC,
  BINARY_ARITHMETIC,
  REDUCE,

  // BLAS2-type
  REDUCE_ROWS,
  REDUCE_COLUMNS,

  // BLAS3-type
  MATRIX_PRODUCT
};

/** @brief Enumeration for identifying the possible operations */
enum token_type
{
  INVALID_TYPE = 0,

  // unary operator
  MINUS_TYPE,
  NEGATE_TYPE,

  // unary expression
  CAST_BOOL_TYPE,
  CAST_CHAR_TYPE,
  CAST_UCHAR_TYPE,
  CAST_SHORT_TYPE,
  CAST_USHORT_TYPE,
  CAST_INT_TYPE,
  CAST_UINT_TYPE,
  CAST_LONG_TYPE,
  CAST_ULONG_TYPE,
  CAST_HALF_TYPE,
  CAST_FLOAT_TYPE,
  CAST_DOUBLE_TYPE,

  ABS_TYPE,
  ACOS_TYPE,
  ASIN_TYPE,
  ATAN_TYPE,
  CEIL_TYPE,
  COS_TYPE,
  COSH_TYPE,
  EXP_TYPE,
  FABS_TYPE,
  FLOOR_TYPE,
  LOG_TYPE,
  LOG10_TYPE,
  SIN_TYPE,
  SINH_TYPE,
  SQRT_TYPE,
  TAN_TYPE,
  TANH_TYPE,
  TRANS_TYPE,

  // binary expression
  ASSIGN_TYPE,
  INPLACE_ADD_TYPE,
  INPLACE_SUB_TYPE,
  ADD_TYPE,
  SUB_TYPE,
  MULT_TYPE,
  DIV_TYPE,
  ELEMENT_ARGFMAX_TYPE,
  ELEMENT_ARGFMIN_TYPE,
  ELEMENT_ARGMAX_TYPE,
  ELEMENT_ARGMIN_TYPE,
  ELEMENT_PROD_TYPE,
  ELEMENT_DIV_TYPE,
  ELEMENT_EQ_TYPE,
  ELEMENT_NEQ_TYPE,
  ELEMENT_GREATER_TYPE,
  ELEMENT_GEQ_TYPE,
  ELEMENT_LESS_TYPE,
  ELEMENT_LEQ_TYPE,
  ELEMENT_POW_TYPE,
  ELEMENT_FMAX_TYPE,
  ELEMENT_FMIN_TYPE,
  ELEMENT_MAX_TYPE,
  ELEMENT_MIN_TYPE,

  //Products
  OUTER_PROD_TYPE,
  MATRIX_PRODUCT_NN_TYPE,
  MATRIX_PRODUCT_TN_TYPE,
  MATRIX_PRODUCT_NT_TYPE,
  MATRIX_PRODUCT_TT_TYPE,

  //Access modifiers
  RESHAPE_TYPE,
  SHIFT_TYPE,
  DIAG_MATRIX_TYPE,
  DIAG_VECTOR_TYPE,
  ACCESS_INDEX_TYPE,


  PAIR_TYPE,

  OPERATOR_FUSE,
  Ssize_tTYPE,
};

struct token
{
  token();
  token(token_family const & _family, token_type const & _type);
  token_family family;
  token_type type;
};

//
std::string to_string(token_type type);
bool is_assignment(token_type op);
bool is_operator(token_type op);
bool is_function(token_type op);
bool is_cast(token_type op);
bool is_indexing(token_type op);
//

enum node_type
{
  INVALID_SUBTYPE = 0,
  COMPOSITE_OPERATOR_TYPE,
  VALUE_SCALAR_TYPE,
  DENSE_ARRAY_TYPE
};

union handle_t
{
  cl_mem cl;
  CUdeviceptr cu;
};

class expression
{
public:
  struct node
  {
    //Constructors
    node();
    node(invalid_node);
    node(scalar const & x);
    node(array_base const & x);
    node(int_t lhs, token op, int_t rhs, numeric_type dtype, tuple const & shape);

    //Common
    node_type type;
    numeric_type dtype;
    tuple shape;
    tuple ld;

    //Type-specific
    union
    {
      //Operator
      struct{
        int lhs;
        token op;
        int rhs;
      }binary_operator;
      //Scalar
      values_holder value;
      //Array
      struct {
        int_t start;
        handle_t handle;
      }array;
    };
  };

  typedef std::vector<node>     data_type;

public:
  expression(node const & lhs, node const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);
  expression(expression const & lhs, node const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);
  expression(node const & lhs, expression const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);
  expression(expression const & lhs, expression const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);

  tuple shape() const;
  int_t dim() const;
  data_type const & data() const;
  std::size_t root() const;
  driver::Context const & context() const;
  numeric_type const & dtype() const;

  node const & operator[](size_t) const;
  node & operator[](size_t);

  expression operator-();
  expression operator!();

private:
  data_type tree_;
  std::size_t root_;
  driver::Context const * context_;
};

//io
std::string to_string(node_type const & f);
std::string to_string(expression::node const & e);
std::ostream & operator<<(std::ostream & os, expression::node const & s_node);
std::string to_string(isaac::expression const & s);

}

#endif
