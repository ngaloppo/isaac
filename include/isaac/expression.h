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
  NEGATE_TYPE,
  // cast
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
  // unary function
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
  //binary operator
  ASSIGN_TYPE,
  ADD_TYPE,
  SUB_TYPE,
  MULT_TYPE,
  DIV_TYPE,
  ELEMENT_ARGFMAX_TYPE,
  ELEMENT_ARGFMIN_TYPE,
  ELEMENT_ARGMAX_TYPE,
  ELEMENT_ARGMIN_TYPE,
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
  //special
  OUTER_PROD_TYPE,
  MATRIX_PRODUCT_NN_TYPE,
  MATRIX_PRODUCT_TN_TYPE,
  MATRIX_PRODUCT_NT_TYPE,
  MATRIX_PRODUCT_TT_TYPE,
  //Access modifiers
  RESHAPE_TYPE,
  DIAG_MATRIX_TYPE,
  DIAG_VECTOR_TYPE
};

struct token
{
  token_family family;
  token_type type;
};

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
    int_t id;
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
  //Initialize
  void init();
  //Accessors
  tuple shape() const;
  int_t dim() const;
  size_t size() const;
  size_t root() const;
  driver::Context const & context() const;
  numeric_type const & dtype() const;
  //Operators
  node const & operator[](size_t) const;
  node & operator[](size_t);
  expression operator-();
  expression operator!();
private:
  data_type tree_;
  size_t root_;
  driver::Context const * context_;
  bool init_;
};
void init(expression & tree);

//to_string
std::string eval(token_type type);
std::string to_string(token const & op);
std::string to_string(expression::node const & e);
std::string to_string(isaac::expression const & s);

//predicates
bool is_operator(token_type op);
bool is_indexing(token_type op);

//depth-first traversal
template<class FUN>
inline void traverse_dfs(expression const & tree, size_t root, FUN const & fun,
                     std::function<bool(size_t)> const & recurse, size_t depth = 0)
{
  expression::node const & node = tree[root];
  if (node.type==COMPOSITE_OPERATOR_TYPE && recurse(root))
  {
    size_t lhs = node.binary_operator.lhs;
    size_t rhs = node.binary_operator.rhs;
    if(tree[lhs].type!=INVALID_SUBTYPE)
      traverse_dfs(tree, lhs, fun, recurse, depth+1);
    if(tree[rhs].type!=INVALID_SUBTYPE)
      traverse_dfs(tree, rhs, fun, recurse, depth+1);
  }
  fun(root, depth);
}

template<class FUN>
inline void traverse_dfs(expression const & tree, size_t root, FUN const & fun)
{
  return traverse_dfs(tree, root, fun,  [](size_t){return true;});
}

template<class FUN>
inline void traverse_dfs(expression const & tree, FUN const & fun)
{
  return traverse_dfs(tree, tree.root(), fun);
}

//breath-first traversal
template<class FUN>
inline void traverse_bfs(expression const & tree, size_t root, FUN const & fun,
                     std::function<bool(size_t)> const & recurse, size_t depth)
{
  fun(root, depth);
  expression::node const & node = tree[root];
  if (node.type==COMPOSITE_OPERATOR_TYPE && recurse(root))
  {
    size_t lhs = node.binary_operator.lhs;
    size_t rhs = node.binary_operator.rhs;
    if(tree[lhs].type!=INVALID_SUBTYPE)
      traverse_bfs(tree, lhs, fun, recurse, depth+1);
    if(tree[rhs].type!=INVALID_SUBTYPE)
      traverse_bfs(tree, rhs, fun, recurse, depth+1);
  }
}

template<class FUN>
inline void traverse_bfs(expression const & tree, size_t root, FUN const & fun)
{
  return traverse_bfs(tree, root, fun,  [](size_t){return true;}, 0);
}

template<class FUN>
inline void traverse_bfs(expression const & tree, FUN const & fun)
{
  return traverse_bfs(tree, tree.root(), fun);
}


}

#endif
