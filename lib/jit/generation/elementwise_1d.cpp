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
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/jit/generation/elementwise_1d.h"
#include "isaac/driver/backend.h"
#include "tools/loop.hpp"
#include "tools/vector_types.hpp"
#include "tools/arguments.hpp"

#include <string>

namespace isaac
{
namespace templates
{

elementwise_1d_parameters::elementwise_1d_parameters(unsigned int _vwidth,
                       unsigned int _group_size, unsigned int _num_groups,
                       fetch_type _fetch) :
      base::parameters_type(_vwidth, _group_size, 1, 1), num_groups(_num_groups), fetch(_fetch)
{
}


int elementwise_1d::is_invalid_impl(driver::Device const &, expression_tree const &) const
{
  if (p_.fetch==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

std::string elementwise_1d::generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  driver::backend_type backend = device.backend();
  kernel_generation_stream stream(backend);

  std::vector<std::size_t> assignments = symbolic::assignments(tree);
  std::vector<std::size_t> assignments_lhs = symbolic::lhs_of(tree, assignments);
  std::vector<std::size_t> assignments_rhs = symbolic::rhs_of(tree, assignments);

  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"vector.h\"" << std::endl; break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << p_.ls0 << "," << p_.ls1 << ",1)))" << std::endl; break;
  }

  stream << "$KERNEL void elementwise_1d" << suffix << "($SIZE_T N, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")";

  stream << "{" << std::endl;
  stream.inc_tab();

  //Open user-provided for-loops
  std::vector<symbolic::sfor*> sfors = symbolic::extract<symbolic::sfor>(tree, symbols);
  for(symbolic::sfor* sym: sfors)
    stream << sym->process("for(int #init ; #end ; #inc)") << std::endl;
  if(sfors.size())
  {
    stream << "{" << std::endl;
    stream.inc_tab();
  }

  element_wise_loop_1D(stream, p_.fetch, p_.vwidth, "i", "N", "$GLOBAL_IDX_0", "$GLOBAL_SIZE_0", device, [&](unsigned int vwidth)
  {
    std::string dtype = append_width("#scalartype",vwidth);

    //Declares register to store results
    for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assignments_lhs, false))
      stream << sym->process(dtype + " #name;") << std::endl;

    //Load to registers
    for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assignments_rhs, false))
      stream << sym->process(dtype + " #name = " + append_width("loadv", vwidth) + "(i);") << std::endl;

    //Compute
    for(size_t idx: assignments)
      for(unsigned int s = 0 ; s < vwidth ; ++s)
         stream << symbols.at(idx)->evaluate({{"leaf", access_vector_type("#name", s, vwidth)}}) << ";" << std::endl;

    //Writes back
    for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assignments_lhs, false))
      for(unsigned int s = 0 ; s < vwidth ; ++s)
          stream << sym->process("at(i+" + tools::to_string(s)+") = " + access_vector_type("#name", s, vwidth) + ";") << std::endl;
  });
  //Close user-provided for-loops
  if(sfors.size()){
    stream.dec_tab();
    stream << "}" << std::endl;
  }

  stream.dec_tab();
  stream << "}" << std::endl;

//  std::cout << stream.str() << std::endl;
  return stream.str();
}

elementwise_1d::elementwise_1d(elementwise_1d_parameters const & parameters,
                               fusion_policy_t fusion_policy) :
    base_impl<elementwise_1d, elementwise_1d_parameters>(parameters, fusion_policy)
{}

elementwise_1d::elementwise_1d(unsigned int simd, unsigned int ls, unsigned int ng,
                               fetch_type fetch, fusion_policy_t bind):
    base_impl<elementwise_1d, elementwise_1d_parameters>(elementwise_1d_parameters(simd,ls,ng,fetch), bind)
{}


std::vector<int_t> elementwise_1d::input_sizes(expression_tree const & expressions) const
{
  return {max(expressions.shape())};
}

void elementwise_1d::enqueue(driver::CommandQueue &, driver::Program const & program, std::string const & suffix, runtime::execution_handler const & control)
{
  expression_tree const & expressions = control.x();
  //Size
  int_t size = input_sizes(expressions)[0];
  //Kernel
  std::string name = "elementwise_1d";
  name += suffix;
  driver::Kernel kernel(program, name.c_str());
  //NDRange
  driver::NDRange global(p_.ls0*p_.num_groups);
  driver::NDRange local(p_.ls0);
  //Arguments
  unsigned int current_arg = 0;
  kernel.setSizeArg(current_arg++, size);
  symbolic::set_arguments(expressions, kernel, current_arg, fusion_policy_);
  control.execution_options().enqueue(program.context(), kernel, global, local);
}


}
}
