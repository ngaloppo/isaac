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
#include "isaac/driver/kernel.h"
#include "isaac/driver/ndrange.h"
#include "isaac/driver/command_queue.h"
#include "isaac/jit/exceptions.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/jit/generation/elementwise_1d.h"
#include "tools/vector_types.hpp"
#include "tools/arguments.hpp"

#include <string>

namespace isaac
{
namespace templates
{

void elementwise_1d::check_valid_impl(driver::Device const &, expression const &) const
{
  if (fetching_policy==FETCH_FROM_LOCAL)
    throw jit::code_generation_error("generated code uses unsupported fetching policy");
}

std::string elementwise_1d::generate_impl(std::string const & suffix, expression const & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  driver::backend_type backend = device.backend();
  genstream stream(backend);

  std::vector<size_t> assignments = symbolic::assignments(tree);
  std::vector<size_t> assignments_lhs = symbolic::lhs_of(tree, assignments);
  std::vector<size_t> assignments_rhs = symbolic::rhs_of(tree, assignments);

  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"vector.h\"" << std::endl; break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << local_size_0 << "," << local_size_1 << ",1)))" << std::endl; break;
  }

  stream << "$KERNEL void elementwise_1d" << suffix << "($SIZE_T N, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")";

  stream << "{" << std::endl;
  stream.inc_tab();

  //Open user-provided for-loops
  for_loop(stream, fetching_policy, simd_width, "i", "N", "$GLOBAL_IDX_0", "$GLOBAL_SIZE_0", [&](size_t simd_width)
  {
    std::string dtype = append_width("#scalartype",simd_width);

    //Declares register to store results
    for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assignments_lhs, false))
      stream << sym->process(dtype + " #name;") << std::endl;

    //Load to registers
    for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assignments_rhs, false))
      stream << sym->process(dtype + " #name = " + append_width("loadv", simd_width) + "(i);") << std::endl;

    //Compute
    for(size_t idx: assignments)
      for(size_t s = 0 ; s < simd_width ; ++s)
         stream << symbols.at(idx)->evaluate({{"leaf", access_vector_type("#name", s, simd_width)}}) << ";" << std::endl;

    //Writes back
    for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assignments_lhs, false))
      for(size_t s = 0 ; s < simd_width ; ++s)
          stream << sym->process("at(i+" + tools::to_string(s)+") = " + access_vector_type("#name", s, simd_width) + ";") << std::endl;
  });

  stream.dec_tab();
  stream << "}" << std::endl;

//  std::cout << stream.str() << std::endl;
  return stream.str();
}

elementwise_1d::elementwise_1d(size_t simd, size_t ls, size_t ng,
                               fetching_policy_type fetch): base(simd, ls, 1), num_groups(ng), fetching_policy(fetch)
{}


std::vector<int_t> elementwise_1d::input_sizes(expression const & expressions) const
{
  return {max(expressions.shape())};
}

void elementwise_1d::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix,
                             expression const & tree, runtime::environment const & opt)
{
  //Size
  int_t size = input_sizes(tree)[0];
  //Kernel
  std::string name = "elementwise_1d";
  name += suffix;
  driver::Kernel kernel(program, name.c_str());
  //NDRange
  driver::NDRange global(local_size_0*num_groups);
  driver::NDRange local(local_size_0);
  //Arguments
  unsigned int current_arg = 0;
  kernel.setSizeArg(current_arg++, size);
  symbolic::set_arguments(tree, kernel, current_arg);
  opt.enqueue(queue.context(), kernel, global, local);
}


}
}
