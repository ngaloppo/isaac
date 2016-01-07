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

#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "../../parse/extract.hpp"

#include "isaac/kernels/symbolic_object.h"
#include "isaac/kernels/parse.h"
#include "isaac/array.h"

namespace isaac
{
namespace templates
{

//Generate
inline std::vector<std::string> kernel_arguments(driver::Device const & device, symbolic::mapping_type const & mappings, expression_tree const & expressions)
{
    std::string kwglobal = Global(device.backend()).get();
    std::string _size_t = size_type(device);
    std::vector<std::string> result;
    for(symbolic::object* obj: extract<symbolic::object>(expressions, mappings))
    {
      if(symbolic::host_scalar* sym = dynamic_cast<symbolic::host_scalar*>(obj))
        result.push_back(sym->process("#scalartype #name"));
      if(symbolic::buffer* sym = dynamic_cast<symbolic::buffer*>(obj))
      {
        result.push_back(kwglobal + " " + sym->process("#scalartype* #pointer"));
        if(sym->hasattr("off")) result.push_back(_size_t + " " + sym->process("#off"));
        if(sym->hasattr("inc0")) result.push_back(_size_t + " " + sym->process("#inc0"));
        if(sym->hasattr("inc1")) result.push_back(_size_t + " " + sym->process("#inc1"));
      }
      if(symbolic::reshape* sym = dynamic_cast<symbolic::reshape*>(obj))
      {
        if(sym->hasattr("new_inc1")) result.push_back(_size_t + " " + sym->process("#new_inc1"));
        if(sym->hasattr("old_inc1")) result.push_back(_size_t + " " + sym->process("#old_inc1"));
      }
    }
    return result;
}

//Enqueue
class set_arguments_functor : public traversal_functor
{
public:
    typedef void result_type;

    set_arguments_functor(symbolic_binder & binder, unsigned int & current_arg, driver::Kernel & kernel)
        : binder_(binder), current_arg_(current_arg), kernel_(kernel)
    {
    }

    void set_arguments(numeric_type dtype, values_holder const & scal) const
    {
        switch(dtype)
        {
        //    case BOOL_TYPE: kernel_.setArg(current_arg_++, scal.bool8); break;
        case CHAR_TYPE: kernel_.setArg(current_arg_++, scal.int8); break;
        case UCHAR_TYPE: kernel_.setArg(current_arg_++, scal.uint8); break;
        case SHORT_TYPE: kernel_.setArg(current_arg_++, scal.int16); break;
        case USHORT_TYPE: kernel_.setArg(current_arg_++, scal.uint16); break;
        case INT_TYPE: kernel_.setArg(current_arg_++, scal.int32); break;
        case UINT_TYPE: kernel_.setArg(current_arg_++, scal.uint32); break;
        case LONG_TYPE: kernel_.setArg(current_arg_++, scal.int64); break;
        case ULONG_TYPE: kernel_.setArg(current_arg_++, scal.uint64); break;
            //    case HALF_TYPE: kernel_.setArg(current_arg_++, scal.float16); break;
        case FLOAT_TYPE: kernel_.setArg(current_arg_++, scal.float32); break;
        case DOUBLE_TYPE: kernel_.setArg(current_arg_++, scal.float64); break;
        default: throw unknown_datatype(dtype);
        }
    }

    void set_arguments(array_base const * a, bool is_assigned) const
    {
        bool is_bound = binder_.bind(a, is_assigned);
        if (is_bound)
        {
            kernel_.setArg(current_arg_++, a->data());
            kernel_.setSizeArg(current_arg_++, a->start());
            for(int_t i = 0 ; i < a->dim() ; i++){
              if(a->shape()[i] > 1)
                kernel_.setSizeArg(current_arg_++, a->stride()[i]);
            }
        }
    }

    void set_arguments(tree_node const & lhs_rhs, bool is_assigned) const
    {
        switch(lhs_rhs.subtype)
        {
        case VALUE_SCALAR_TYPE:   return set_arguments(lhs_rhs.dtype, lhs_rhs.scalar);
        case DENSE_ARRAY_TYPE:    return set_arguments(lhs_rhs.array, is_assigned);
        case PLACEHOLDER_TYPE: return;
        default: throw std::runtime_error("Unrecognized type family");
        }
    }

    void operator()(isaac::expression_tree const & expression, size_t root_idx, leaf_t leaf_t) const
    {
        expression_tree::node const & node = expression.tree()[root_idx];
        if (leaf_t==LHS_NODE_TYPE && node.lhs.subtype != COMPOSITE_OPERATOR_TYPE)
          set_arguments(node.lhs, detail::is_assignment(node.op));
        else if (leaf_t==RHS_NODE_TYPE && node.rhs.subtype != COMPOSITE_OPERATOR_TYPE)
          set_arguments(node.rhs, false);
        if(leaf_t==PARENT_NODE_TYPE && node.op.type == RESHAPE_TYPE)
        {
          tuple const & new_shape = node.shape;
          tuple const & old_shape = node.lhs.subtype==DENSE_ARRAY_TYPE?node.lhs.array->shape():expression.tree()[node.lhs.index].shape;
          for(unsigned int i = 1 ; i < new_shape.size() ; ++i)
            kernel_.setSizeArg(current_arg_++, new_shape[i]);
          for(unsigned int i = 1 ; i < old_shape.size() ; ++i)
            kernel_.setSizeArg(current_arg_++, old_shape[i]);
          }
    }


private:
    symbolic_binder & binder_;
    unsigned int & current_arg_;
    driver::Kernel & kernel_;
};

inline void set_arguments(expression_tree const & expression, driver::Kernel & kernel, unsigned int & current_arg, binding_policy_t binding_policy)
{
    std::unique_ptr<symbolic_binder> binder;
    if (binding_policy==BIND_SEQUENTIAL)
        binder.reset(new bind_sequential());
    else
        binder.reset(new bind_independent());
    traverse(expression, expression.root(), set_arguments_functor(*binder, current_arg, kernel), true);

}

}
}
