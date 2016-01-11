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

#ifndef ISAAC_KERNELS_PARSE_SET_ARGUMENTS_HPP
#define ISAAC_KERNELS_PARSE_SET_ARGUMENTS_HPP

#include <cstring>

#include "isaac/kernels/symbolic_object.h"
#include "isaac/kernels/parse.h"
#include "traverse.hpp"
#include "isaac/array.h"

namespace isaac
{

namespace detail
{

class set_arguments
{
public:
    set_arguments(symbolic_binder & binder, unsigned int & current_arg, expression_tree const & expression, driver::Kernel & kernel)
        : binder_(binder), current_arg_(current_arg), expression_(expression), kernel_(kernel)
    {
    }

    void setArg(tree_node const & lhs_rhs, bool is_assigned) const
    {
        switch(lhs_rhs.subtype)
        {
        case VALUE_SCALAR_TYPE:
          return kernel_.setArg(current_arg_++,value_scalar(lhs_rhs.scalar,lhs_rhs.dtype));
        case DENSE_ARRAY_TYPE:
        {
          array_base const * array = lhs_rhs.array;
          bool is_bound = binder_.bind(array, is_assigned);
          if (is_bound)
          {
              kernel_.setArg(current_arg_++, array->data());
              kernel_.setSizeArg(current_arg_++, array->start());
              for(int_t i = 0 ; i < array->dim() ; i++)
              {
                if(array->shape()[i] > 1)
                  kernel_.setSizeArg(current_arg_++, array->stride()[i]);
              }
          }
        }
        case PLACEHOLDER_TYPE:
          return;
        default:
          throw std::runtime_error("Unrecognized type family");
        }
    }

    void operator()(size_t root_idx, leaf_t leaf) const
    {
        expression_tree::node const & node = expression_.tree()[root_idx];
        if (leaf==LHS_NODE_TYPE && node.lhs.subtype != COMPOSITE_OPERATOR_TYPE)
          setArg(node.lhs, detail::is_assignment(node.op.type));
        else if (leaf==RHS_NODE_TYPE && node.rhs.subtype != COMPOSITE_OPERATOR_TYPE)
          setArg(node.rhs, false);
        if(leaf==PARENT_NODE_TYPE && node.op.type == RESHAPE_TYPE)
        {
          tuple const & new_shape = node.shape;
          int_t current = 1;
          for(unsigned int i = 1 ; i < new_shape.size() ; ++i){
            current *= new_shape[i-1];
            if(new_shape[i] > 1)
              kernel_.setSizeArg(current_arg_++, current);
          }

          tuple const & old_shape = node.lhs.subtype==DENSE_ARRAY_TYPE?node.lhs.array->shape():expression_.tree()[node.lhs.index].shape;
          current = 1;
          for(unsigned int i = 1 ; i < old_shape.size() ; ++i){
            current *= old_shape[i-1];
            if(old_shape[i] > 1)
              kernel_.setSizeArg(current_arg_++, current);
          }
        }
    }


private:
    symbolic_binder & binder_;
    unsigned int & current_arg_;
    expression_tree const & expression_;
    driver::Kernel & kernel_;
};


}

inline void set_arguments(expression_tree const & expression, driver::Kernel & kernel, unsigned int & current_arg, binding_policy_t binding_policy)
{
  std::unique_ptr<symbolic_binder> binder;
  if (binding_policy==BIND_SEQUENTIAL)
      binder.reset(new bind_sequential());
  else
      binder.reset(new bind_independent());
  _traverse(expression, detail::set_arguments(*binder, current_arg, expression, kernel));
}

}

#endif
