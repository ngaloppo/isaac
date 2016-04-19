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
#include <iostream>
#include "isaac/driver/command_queue.h"
#include "isaac/jit/exceptions.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/ndrange.h"
#include "isaac/jit/generation/elementwise_2d.h"
#include "isaac/jit/syntax/engine/process.h"
#include "tools/arguments.hpp"
#include "tools/vector_types.hpp"


namespace isaac
{
namespace templates
{

void elementwise_2d::check_valid_impl(driver::Device const &, expression_tree const  &) const
{
  if (simd_width>1)
    throw jit::code_generation_error("generated code uses invalid simd width");
  if(fetching_policy==FETCH_FROM_LOCAL)
    throw jit::code_generation_error("generated code uses unsupported fetching policy");
}

std::string elementwise_2d::generate_impl(std::string const & suffix, expression_tree const  & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  std::string init0, upper_bound0, inc0, init1, upper_bound1, inc1;
  driver::backend_type backend = device.backend();
  genstream stream(backend);

  std::vector<std::size_t> assigned = symbolic::find(tree, [&](expression_tree::node const & node){return node.type==COMPOSITE_OPERATOR_TYPE && is_assignment(node.binary_operator.op.type);});
  std::vector<std::size_t> assigned_left;
  std::vector<std::size_t> assigned_right;
  for(std::size_t idx: assigned){
    assigned_left.push_back(tree[idx].binary_operator.lhs);
    assigned_right.push_back(tree[idx].binary_operator.rhs);
  }
  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"vector.h\"" << std::endl; break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << local_size_0 << "," << local_size_1 << ",1)))" << std::endl; break;
  }

  stream << "$KERNEL void elementwise_2d" << suffix << "($SIZE_T M, $SIZE_T N, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for_loop(stream, fetching_policy, 1, "i", "M",  "$GLOBAL_IDX_0", "$GLOBAL_SIZE_0", [&](size_t)
  {
    for_loop(stream, fetching_policy, 1, "j", "N",  "$GLOBAL_IDX_1", "$GLOBAL_SIZE_1", [&](size_t)
    {
      //Declares register to store results
      for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assigned_left, false))
        stream << sym->process("#scalartype #name;") << std::endl;

      //Load to registers
      for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assigned_right, false))
        stream << sym->process("#scalartype #name = at(i, j);") << std::endl;

      for(std::size_t idx: assigned)
        stream << symbols.at(idx)->evaluate({{"leaf", "#name"}}) << ";" << std::endl;

      //Writes back
      for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, assigned_left, false))
        stream << sym->process("at(i, j) = #name;") << std::endl;
    });
  });

  stream.dec_tab();
  stream << "}" << std::endl;

  return stream.str();
}

elementwise_2d::elementwise_2d(size_t simd, size_t ls0, size_t ls1,
                               size_t ng0, size_t ng1, fetching_policy_type fetch):
    base(simd, ls0, ls1), num_groups_0(ng0), num_groups_1(ng1), fetching_policy(fetch)
{}

std::vector<int_t> elementwise_2d::input_sizes(expression_tree const  & tree) const
{
  return tree.shape();
}

void elementwise_2d::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix,
                             expression_tree const & tree, runtime::environment const & opt)
{
  std::string name = "elementwise_2d";
  name +=suffix;
  driver::Kernel kernel(program, name.c_str());
  driver::NDRange global(local_size_0*num_groups_0, local_size_1*num_groups_1);
  driver::NDRange local(local_size_0, local_size_1);
  unsigned int current_arg = 0;
  std::vector<int_t> MN = input_sizes(tree);
  kernel.setSizeArg(current_arg++, MN[0]);
  kernel.setSizeArg(current_arg++, MN[1]);
  symbolic::set_arguments(tree, kernel, current_arg);
  opt.enqueue(queue.context(), kernel, global, local);
}

}
}
