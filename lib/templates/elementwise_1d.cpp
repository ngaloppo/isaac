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

#include <iostream>
#include <cstring>
#include <algorithm>

#include "isaac/symbolic/engine/process.h"

#include "isaac/templates/elementwise_1d.h"
#include "isaac/driver/backend.h"

#include "tools/loop.hpp"
#include "tools/vector_types.hpp"
#include "tools/arguments.hpp"
#include "isaac/symbolic/expression/io.h"

#include <string>

namespace isaac
{
namespace templates
{

elementwise_1d_parameters::elementwise_1d_parameters(unsigned int _simd_width,
                       unsigned int _group_size, unsigned int _num_groups,
                       fetching_policy_type _fetching_policy) :
      base::parameters_type(_simd_width, _group_size, 1, 1), num_groups(_num_groups), fetching_policy(_fetching_policy)
{
}


int elementwise_1d::is_invalid_impl(driver::Device const &, expression_tree const &) const
{
  if (p_.fetching_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

std::string elementwise_1d::generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  driver::backend_type backend = device.backend();
  kernel_generation_stream stream(backend);

  expression_tree::data_type const & data = tree.data();
  std::vector<std::size_t> sfors = symbolic::filter(tree, [](expression_tree::node const & node){return node.op.type==SFOR_TYPE;});
  size_t root = tree.root();
  if(sfors.size())
      root = data[sfors.back()].lhs.index;

  std::vector<std::size_t> assigned = symbolic::filter(tree, root, PARENT_NODE_TYPE, [](expression_tree::node const & node){return is_assignment(node.op.type);});

  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"helper_math.h\"" << std::endl; break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl; break;
  }

  stream << "$KERNEL void elementwise_1d" << suffix << "($SIZE_T N, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")";

  stream << "{" << std::endl;
  stream.inc_tab();

  element_wise_loop_1D(stream, p_.fetching_policy, p_.simd_width, "i", "N", "$GLOBAL_IDX_0", "$GLOBAL_SIZE_0", device, [&](unsigned int simd_width)
  {
    std::string dtype = append_width("#scalartype",simd_width);

    //Open user-provided for-loops
    for(unsigned int i = 0 ; i < sfors.size() ; ++i)
    {
      std::string info[3];
      int idx =  sfors[i];
      for(int i = 0 ; i < 2 ; ++i){
          idx = data[idx].rhs.index;
          info[i] = symbols.at({idx, LHS_NODE_TYPE})->process("#name");

      }
      info[2] = symbols.at({idx, RHS_NODE_TYPE})->process("#name");
      info[0] = info[0].substr(1, info[0].size()-2);
      stream << "for(int " << info[0] << " ; " << info[1] << "; " << info[2] << ")" << std::endl;
    }

    if(sfors.size()){
      stream << "{" << std::endl;
      stream.inc_tab();
    }

    //Declares register to store results
    for(symbolic::array* sym: symbolic::extract<symbolic::array>(tree, symbols, assigned, LHS_NODE_TYPE))
      stream << sym->process(dtype + " #name;") << std::endl;

    //Load to registers
    for(symbolic::array* sym: symbolic::extract<symbolic::array>(tree, symbols, assigned, RHS_NODE_TYPE))
    {
      if(simd_width==1)
        stream << sym->process(dtype + " #name = at(i);") << std::endl;
      if(simd_width==2)
        stream << sym->process(dtype + " #name = (#scalartype2)(at(i), at(i+1));") << std::endl;
      if(simd_width==4)
        stream << sym->process(dtype + " #name = (#scalartype4)(at(i), at(i+1), at(i+2), at(i+3));") << std::endl;
    }

    //Compute
    for(std::size_t idx: assigned)
      for(unsigned int s = 0 ; s < simd_width ; ++s)
         stream << symbols.at({idx, PARENT_NODE_TYPE})->evaluate(access_vector_type("#name", s, simd_width)) << ";" << std::endl;


    //Writes back
    for(symbolic::array* sym: symbolic::extract<symbolic::array>(tree, symbols, assigned, LHS_NODE_TYPE))
      for(unsigned int s = 0 ; s < simd_width ; ++s)
          stream << sym->process("at(i+" + tools::to_string(s)+") = " + access_vector_type("#name", s, simd_width) + ";") << std::endl;

    //Close user-provided for-loops
    if(sfors.size())
    {
      stream.dec_tab();
      stream << "}" << std::endl;
    }

  });

  stream.dec_tab();
  stream << "}" << std::endl;

  return stream.str();
}

elementwise_1d::elementwise_1d(elementwise_1d_parameters const & parameters,
                               fusion_policy_t fusion_policy) :
    base_impl<elementwise_1d, elementwise_1d_parameters>(parameters, fusion_policy)
{}

elementwise_1d::elementwise_1d(unsigned int simd, unsigned int ls, unsigned int ng,
                               fetching_policy_type fetch, fusion_policy_t bind):
    base_impl<elementwise_1d, elementwise_1d_parameters>(elementwise_1d_parameters(simd,ls,ng,fetch), bind)
{}


std::vector<int_t> elementwise_1d::input_sizes(expression_tree const & expressions) const
{
  return {max(expressions.shape())};
}

void elementwise_1d::enqueue(driver::CommandQueue &, driver::Program const & program, std::string const & suffix, base &, execution_handler const & control)
{
  expression_tree const & expressions = control.x();
  //Size
  int_t size = input_sizes(expressions)[0];
  //Kernel
  std::string name = "elementwise_1d";
  name += suffix;
  driver::Kernel kernel(program, name.c_str());
  //NDRange
  driver::NDRange global(p_.local_size_0*p_.num_groups);
  driver::NDRange local(p_.local_size_0);
  //Arguments
  unsigned int current_arg = 0;
  kernel.setSizeArg(current_arg++, size);
  symbolic::set_arguments(expressions, kernel, current_arg, fusion_policy_);
  control.execution_options().enqueue(program.context(), kernel, global, local);
}


}
}
