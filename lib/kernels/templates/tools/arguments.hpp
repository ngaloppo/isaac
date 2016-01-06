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
        if(sym->hasattr("start")) result.push_back(_size_t + " " + sym->process("#start"));
        if(sym->hasattr("stride")) result.push_back(_size_t + " " + sym->process("#stride"));
        if(sym->hasattr("ld")) result.push_back(_size_t + " " + sym->process("#ld"));
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
        case VALUE_SCALAR_TYPE:   return set_arguments(lhs_rhs.dtype, lhs_rhs.vscalar);
        case DENSE_ARRAY_TYPE:    return set_arguments(lhs_rhs.array, is_assigned);
        case FOR_LOOP_INDEX_TYPE: return;
        default: throw std::runtime_error("Unrecognized type family");
        }
    }

    void operator()(isaac::expression_tree const & expression_tree, size_t root_idx, leaf_t leaf_t) const
    {
        expression_tree::node const & root_node = expression_tree.tree()[root_idx];
        if (leaf_t==LHS_NODE_TYPE && root_node.lhs.subtype != COMPOSITE_OPERATOR_TYPE)
            set_arguments(root_node.lhs, detail::is_assignment(root_node.op));
        else if (leaf_t==RHS_NODE_TYPE && root_node.rhs.subtype != COMPOSITE_OPERATOR_TYPE)
            set_arguments(root_node.rhs, false);
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
