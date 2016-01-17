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
#include "isaac/templates/elementwise_2d.h"
#include "isaac/symbolic/expression/io.h"
#include "isaac/templates/keywords.h"

#include "tools/arguments.hpp"
#include "tools/loop.hpp"
#include "tools/vector_types.hpp"

#include "../parse/set_arguments.hpp"
#include "../parse/filter.hpp"

namespace isaac
{
namespace templates
{

elementwise_2d_parameters::elementwise_2d_parameters(unsigned int _simd_width,
                          unsigned int _local_size_0, unsigned int _local_size_1,
                          unsigned int _num_groups_0, unsigned int _num_groups_1,
                          fetching_policy_type _fetching_policy) : base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1), num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetching_policy(_fetching_policy){ }



int elementwise_2d::is_invalid_impl(driver::Device const &, expression_tree const  &) const
{
  if (p_.simd_width>1)
    return TEMPLATE_INVALID_SIMD_WIDTH;
  if(p_.fetching_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

std::string elementwise_2d::generate_impl(std::string const & suffix, expression_tree const  & expressions, driver::Device const & device, symbolic::mapping_type const & mappings) const
{
  kernel_generation_stream stream;
  std::string _size_t = size_type(device);
  std::string init0, upper_bound0, inc0, init1, upper_bound1, inc1;
  std::string data_type = append_width("#scalartype",p_.simd_width);
  driver::backend_type backend = device.backend();
  std::vector<std::size_t> assigned = filter(expressions, [](expression_tree::node const & node){return is_assignment(node.op.type);});

  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"helper_math.h\"" << std::endl; break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl; break;
  }

  stream << KernelPrefix(backend) << " void elementwise_2d" << suffix << "(" << _size_t << " M, " << _size_t << " N, " << tools::join(kernel_arguments(device, mappings, expressions), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();


  fetching_loop_info(p_.fetching_policy, "M", stream, init0, upper_bound0, inc0,  GlobalIdx0(backend).get(), GlobalSize0(backend).get(), device);
  stream << "for(" << _size_t << " i = " << init0 << "; i < " << upper_bound0 << "; i += " << inc0 << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  fetching_loop_info(p_.fetching_policy, "N", stream, init1, upper_bound1, inc1, GlobalIdx1(backend).get(), GlobalSize1(backend).get(), device);
  stream << "for(" << _size_t << " j = " << init1 << "; j < " << upper_bound1 << "; j += " << inc1 << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  //Declares register to store results
  for(symbolic::array* sym: extract<symbolic::array>(expressions, mappings, assigned, LHS_NODE_TYPE))
    stream << sym->process("#scalartype #name;") << std::endl;

  //Load to registers
  for(symbolic::array* sym: extract<symbolic::array>(expressions, mappings, assigned, RHS_NODE_TYPE))
    stream << sym->process("#scalartype #name = at(i, j);") << std::endl;

  for(std::size_t idx: assigned)
    stream << mappings.at({idx, PARENT_NODE_TYPE})->evaluate("#name") << ";" << std::endl;

  //Writes back
  for(symbolic::array* sym: extract<symbolic::buffer>(expressions, mappings, assigned, LHS_NODE_TYPE))
    stream << sym->process("at(i, j) = #name;") << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

//  std::cout << stream.str() << std::endl;

  return stream.str();
}

elementwise_2d::elementwise_2d(parameters_type const & parameters, binding_policy_t binding_policy) :
  base_impl<elementwise_2d, elementwise_2d_parameters>(parameters, binding_policy){ }

elementwise_2d::elementwise_2d(unsigned int simd, unsigned int ls1, unsigned int ls2,
                               unsigned int ng1, unsigned int ng2, fetching_policy_type fetch,
                               binding_policy_t bind):
    base_impl<elementwise_2d, elementwise_2d_parameters>(elementwise_2d_parameters(simd, ls1, ls2, ng1, ng2, fetch), bind)
{}

std::vector<int_t> elementwise_2d::input_sizes(expression_tree const  & expression) const
{
  std::pair<int_t, int_t> size = matrix_size(expression.data(), lhs_most(expression.data(), expression.root()));
  return {size.first, size.second};
}

void elementwise_2d::enqueue(driver::CommandQueue & /*queue*/, driver::Program const & program, std::string const & suffix, base &, execution_handler const & control)
{
  expression_tree const  & expressions = control.x();
  std::string name = "elementwise_2d";
  name +=suffix;
  driver::Kernel kernel(program, name.c_str());
  driver::NDRange global(p_.local_size_0*p_.num_groups_0, p_.local_size_1*p_.num_groups_1);
  driver::NDRange local(p_.local_size_0, p_.local_size_1);
  unsigned int current_arg = 0;
  std::vector<int_t> MN = input_sizes(expressions);
  kernel.setSizeArg(current_arg++, MN[0]);
  kernel.setSizeArg(current_arg++, MN[1]);
  set_arguments(expressions, kernel, current_arg, binding_policy_);

  control.execution_options().enqueue(program.context(), kernel, global, local);
}

}
}
