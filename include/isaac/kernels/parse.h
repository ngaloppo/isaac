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

#ifndef ISAAC_BACKEND_PARSE_H
#define ISAAC_BACKEND_PARSE_H

#include <set>
#include "isaac/kernels/symbolic_object.h"
#include "isaac/kernels/binder.h"
#include "isaac/symbolic/expression/expression.h"

namespace isaac
{

namespace detail
{

  bool is_scalar_reduce_1d(expression_tree::node const & node);
  bool is_reduce_2d(expression_tree::node const & node);
  bool is_assignment(operation_type op);
  bool is_elementwise_operator(op_element const & op);
  bool is_elementwise_function(op_element const & op);
  bool is_cast(op_element const & op);
}

}
#endif
