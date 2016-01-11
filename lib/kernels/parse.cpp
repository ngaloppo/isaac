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

#include <cstring>

#include "isaac/tools/cpp/string.hpp"

#include "isaac/array.h"
#include "isaac/kernels/parse.h"
#include "isaac/exception/operation_not_supported.h"

namespace isaac
{

namespace detail
{

  bool is_scalar_reduce_1d(expression_tree::node const & node)
  {
    return node.op.type_family==VECTOR_DOT_TYPE_FAMILY;
  }

  bool is_reduce_2d(expression_tree::node const & node)
  {
    return node.op.type_family==ROWS_DOT_TYPE_FAMILY
        || node.op.type_family==COLUMNS_DOT_TYPE_FAMILY;
  }

  bool is_assignment(operation_type op)
  {
      return op== ASSIGN_TYPE
              || op== INPLACE_ADD_TYPE
              || op== INPLACE_SUB_TYPE;
  }

  bool is_elementwise_operator(op_element const & op)
  {
    return is_assignment(op.type)
        || op.type== ADD_TYPE
        || op.type== SUB_TYPE
        || op.type== ELEMENT_PROD_TYPE
        || op.type== ELEMENT_DIV_TYPE
        || op.type== MULT_TYPE
        || op.type== DIV_TYPE
        || op.type== ELEMENT_EQ_TYPE
        || op.type== ELEMENT_NEQ_TYPE
        || op.type== ELEMENT_GREATER_TYPE
        || op.type== ELEMENT_LESS_TYPE
        || op.type== ELEMENT_GEQ_TYPE
        || op.type== ELEMENT_LEQ_TYPE ;
  }

  bool bypass(op_element const & op)
  {
        return op.type == TRANS_TYPE;
  }

  bool is_cast(op_element const & op)
  {
        return op.type== CAST_BOOL_TYPE
            || op.type== CAST_CHAR_TYPE
            || op.type== CAST_UCHAR_TYPE
            || op.type== CAST_SHORT_TYPE
            || op.type== CAST_USHORT_TYPE
            || op.type== CAST_INT_TYPE
            || op.type== CAST_UINT_TYPE
            || op.type== CAST_LONG_TYPE
            || op.type== CAST_ULONG_TYPE
            || op.type== CAST_FLOAT_TYPE
            || op.type== CAST_DOUBLE_TYPE
            ;
  }

  bool is_elementwise_function(op_element const & op)
  {
    return is_cast(op)
        || op.type== ABS_TYPE
        || op.type== ACOS_TYPE
        || op.type== ASIN_TYPE
        || op.type== ATAN_TYPE
        || op.type== CEIL_TYPE
        || op.type== COS_TYPE
        || op.type== COSH_TYPE
        || op.type== EXP_TYPE
        || op.type== FABS_TYPE
        || op.type== FLOOR_TYPE
        || op.type== LOG_TYPE
        || op.type== LOG10_TYPE
        || op.type== SIN_TYPE
        || op.type== SINH_TYPE
        || op.type== SQRT_TYPE
        || op.type== TAN_TYPE
        || op.type== TANH_TYPE

        || op.type== ELEMENT_POW_TYPE
        || op.type== ELEMENT_FMAX_TYPE
        || op.type== ELEMENT_FMIN_TYPE
        || op.type== ELEMENT_MAX_TYPE
        || op.type== ELEMENT_MIN_TYPE;

  }
}

}
