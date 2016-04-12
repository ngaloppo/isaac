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
#include "isaac/driver/backend.h"
#include "isaac/driver/context.h"
#include "isaac/driver/command_queue.h"
#include "isaac/driver/event.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/ndrange.h"
#include "isaac/driver/buffer.h"
#include "isaac/jit/syntax/expression/operations.h"
#include "isaac/tools/cpp/tuple.hpp"
#include "isaac/scalar.h"
#include <memory>
#include <iostream>

namespace isaac
{

class array_base;

struct invalid_node{};

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

class expression_tree
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
  expression_tree(node const & lhs, node const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);
  expression_tree(expression_tree const & lhs, node const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);
  expression_tree(node const & lhs, expression_tree const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);
  expression_tree(expression_tree const & lhs, expression_tree const & rhs, token const & op, driver::Context const * context, numeric_type const & dtype, tuple const & shape);

  tuple shape() const;
  int_t dim() const;
  data_type const & data() const;
  std::size_t root() const;
  driver::Context const & context() const;
  numeric_type const & dtype() const;

  node const & operator[](size_t) const;
  node & operator[](size_t);

  expression_tree operator-();
  expression_tree operator!();

private:
  data_type tree_;
  std::size_t root_;
  driver::Context const * context_;
};

//io
std::string to_string(node_type const & f);
std::string to_string(expression_tree::node const & e);
std::ostream & operator<<(std::ostream & os, expression_tree::node const & s_node);
std::string to_string(isaac::expression_tree const & s);

}

#endif
