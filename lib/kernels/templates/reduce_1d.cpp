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
#include "isaac/kernels/templates/reduce_1d.h"
#include "isaac/kernels/keywords.h"

#include "../parse/extract.hpp"
#include "../parse/set_arguments.hpp"
#include "../parse/filter.hpp"

#include "tools/loop.hpp"
#include "tools/reductions.hpp"
#include "tools/vector_types.hpp"
#include "tools/arguments.hpp"

#include <string>


namespace isaac
{
namespace templates
{
reduce_1d_parameters::reduce_1d_parameters(unsigned int _simd_width,
                     unsigned int _group_size, unsigned int _num_groups,
                     fetching_policy_type _fetching_policy) : base::parameters_type(_simd_width, _group_size, 1, 2), num_groups(_num_groups), fetching_policy(_fetching_policy)
{ }

unsigned int reduce_1d::lmem_usage(expression_tree const  & x) const
{
  numeric_type numeric_t= lhs_most(x.tree(), x.root()).lhs.dtype;
  return p_.local_size_0*size_of(numeric_t);
}

int reduce_1d::is_invalid_impl(driver::Device const &, expression_tree const  &) const
{
  if (p_.fetching_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

unsigned int reduce_1d::temporary_workspace(expression_tree const &) const
{
    if(p_.num_groups > 1)
      return p_.num_groups;
    return 0;
}

inline void reduce_1d::reduce_1d_local_memory(kernel_generation_stream & stream, unsigned int size, std::vector<symbolic::reduce_1d*> exprs,
                                   std::string const & buf_str, std::string const & buf_value_str, driver::backend_type backend) const
{
  stream << "#pragma unroll" << std::endl;
  stream << "for(unsigned int stride = " << size/2 << "; stride > 0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << LocalBarrier(backend) << ";" << std::endl;
  stream << "if (lid <  stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (symbolic::reduce_1d* rd : exprs)
    if (is_index_reduction(rd->op()))
      compute_index_reduce_1d(stream, rd->process(buf_str+"[lid]"), rd->process(buf_str+"[lid+stride]")
                              , rd->process(buf_value_str+"[lid]"), rd->process(buf_value_str+"[lid+stride]"),
                              rd->op());
    else
      compute_reduce_1d(stream, rd->process(buf_str+"[lid]"), rd->process(buf_str+"[lid+stride]"), rd->op());
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
}

std::string reduce_1d::generate_impl(std::string const & suffix, expression_tree const  & expressions, driver::Device const & device, symbolic::mapping_type const & mapping) const
{
  kernel_generation_stream stream;

  std::vector<symbolic::reduce_1d*> reductions = extract<symbolic::reduce_1d>(expressions, mapping);
  driver::backend_type backend = device.backend();
  std::string _size_t = size_type(device);
  std::string _global =  Global(backend).get();

  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;

  auto unroll_tmp = [&]()
  {
      unsigned int offset = 0;
      for(symbolic::reduce_1d* rd: reductions)
      {
        numeric_type dtype = lhs_most(expressions.tree(), expressions.root()).lhs.dtype;
        std::string sdtype = to_string(dtype);
        if (is_index_reduction(rd->op()))
        {
          stream << rd->process(_global + " uint* #name_temp = (" + _global + " uint *)(tmp + " + tools::to_string(offset) + ");");
          offset += 4*p_.num_groups;
          stream << rd->process(_global + " " + sdtype + "* #name_temp_value = (" + _global + " " + sdtype + "*)(tmp + " + tools::to_string(offset) + ");");
          offset += size_of(dtype)*p_.num_groups;
        }
        else{
          stream << rd->process( _global + " " + sdtype + "* #name_temp = (" + _global + " " + sdtype + "*)(tmp + " + tools::to_string(offset) + ");");
          offset += size_of(dtype)*p_.num_groups;
        }
      }
  };

  /* ------------------------
   * First Kernel
   * -----------------------*/
  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"helper_math.h\"" << std::endl; break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << ",1,1)))" << std::endl; break;
  }

  stream << KernelPrefix(backend) << " void " << name[0] << "(" << _size_t << " N, " << _global << " char* tmp," << tools::join(kernel_arguments(device, mapping, expressions), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  unroll_tmp();

  stream << "unsigned int lid = " <<LocalIdx0(backend) << ";" << std::endl;
  stream << "unsigned int gid = " <<GlobalIdx0(backend) << ";" << std::endl;
  stream << "unsigned int gpid = " <<GroupIdx0(backend) << ";" << std::endl;
  stream << "unsigned int gsize = " <<GlobalSize0(backend) << ";" << std::endl;

  for(symbolic::reduce_1d* rd: reductions)
  {
    if (is_index_reduction(rd->op()))
    {
      stream << rd->process(Local(backend).get() + " #scalartype #name_buf_value[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << rd->process("#scalartype #name_acc_value = " + neutral_element(rd->op(), backend, "#scalartype") + ";") << std::endl;
      stream << rd->process(Local(backend).get() + " unsigned int #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << rd->process("unsigned int #name_acc = 0;") << std::endl;
    }
    else
    {
      stream << rd->process(Local(backend).get() + " #scalartype #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << rd->process("#scalartype #name_acc = " + neutral_element(rd->op(), backend, "#scalartype") + ";") << std::endl;
    }
  }


  element_wise_loop_1D(stream, p_.fetching_policy, p_.simd_width, "i", "N", GlobalIdx0(backend).get(), GlobalSize0(backend).get(), device, [&](unsigned int simd_width)
  {
    std::string i = (simd_width==1)?"i*#stride":"i";

    //Fetch vector entry
    std::set<std::string> fetched;
     for (symbolic::reduce_1d* rd : reductions)
       for(symbolic::buffer* array: extract<symbolic::buffer>(expressions, mapping, rd->index(), PARENT_NODE_TYPE))
          if(fetched.insert(array->process("#name")).second)
           stream << array->process(append_width("#scalartype",simd_width) + " #name = " + vload(simd_width,"#scalartype",i,"#pointer","#stride",backend)+";") << std::endl;


    //Update accumulators
    for (symbolic::reduce_1d* rd : reductions)
      for (unsigned int s = 0; s < simd_width; ++s)
      {
        std::string value = mapping.at({rd->index(), LHS_NODE_TYPE})->evaluate(access_vector_type("#name", s, simd_width));
        if (is_index_reduction(rd->op()))
          compute_index_reduce_1d(stream, rd->process("#name_acc"),  "i*" + tools::to_string(simd_width) + "+" + tools::to_string(s), rd->process("#name_acc_value"), value,rd->op());
        else
          compute_reduce_1d(stream, rd->process("#name_acc"), value,rd->op());
      }
  });

  //Fills local memory
  for(symbolic::reduce_1d* rd: reductions)
  {
    if (is_index_reduction(rd->op()))
      stream << rd->process("#name_buf_value[lid] = #name_acc_value;") << std::endl;
    stream << rd->process("#name_buf[lid] = #name_acc;") << std::endl;
  }

  //Reduce local memory
  reduce_1d_local_memory(stream, p_.local_size_0, reductions, "#name_buf", "#name_buf_value", backend);

  //Write to temporary buffers
  stream << "if (lid==0)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  for(symbolic::reduce_1d* rd: reductions)
  {
    if (is_index_reduction(rd->op()))
      stream << rd->process("#name_temp_value[gpid] = #name_buf_value[0];") << std::endl;
    stream << rd->process("#name_temp[gpid] = #name_buf[0];") << std::endl;
  }
  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;

  /* ------------------------
   * Second kernel
   * -----------------------*/



  stream << KernelPrefix(backend) << " void " << name[1] << "(" << _size_t << " N, " << _global << " char* tmp, " << tools::join(kernel_arguments(device, mapping, expressions), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  unroll_tmp();

  stream << "unsigned int lid = " <<LocalIdx0(backend) << ";" << std::endl;
  stream << "unsigned int lsize = " <<LocalSize0(backend) << ";" << std::endl;

  for (symbolic::reduce_1d* rd: reductions)
  {
    if (is_index_reduction(rd->op()))
    {
      stream << rd->process(Local(backend).get() + " unsigned int #name_buf[" + tools::to_string(p_.local_size_0) + "];");
      stream << rd->process("unsigned int #name_acc = 0;") << std::endl;
      stream << rd->process(Local(backend).get() + " #scalartype #name_buf_value[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << rd->process("#scalartype #name_acc_value = " + neutral_element(rd->op(), backend, "#scalartype") + ";");
    }
    else
    {
      stream << rd->process(Local(backend).get() + " #scalartype #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << rd->process("#scalartype #name_acc = " + neutral_element(rd->op(), backend, "#scalartype") + ";");
    }
  }

  stream << "for(unsigned int i = lid; i < " << p_.num_groups << "; i += lsize)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  for (symbolic::reduce_1d* rd: reductions)
    if (is_index_reduction(rd->op()))
      compute_index_reduce_1d(stream, rd->process("#name_acc"), rd->process("#name_temp[i]"), rd->process("#name_acc_value"),rd->process("#name_temp_value[i]"),rd->op());
    else
      compute_reduce_1d(stream, rd->process("#name_acc"), rd->process("#name_temp[i]"), rd->op());

  stream.dec_tab();
  stream << "}" << std::endl;

  for(symbolic::reduce_1d* rd: reductions)
  {
    if (is_index_reduction(rd->op()))
      stream << rd->process("#name_buf_value[lid] = #name_acc_value;") << std::endl;
    stream << rd->process("#name_buf[lid] = #name_acc;") << std::endl;
  }


  //Reduce and write final result
  reduce_1d_local_memory(stream, p_.local_size_0, reductions, "#name_buf", "#name_buf_value", backend);

  stream << "if (lid==0)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << mapping.at({expressions.root(), PARENT_NODE_TYPE})->evaluate("#name_buf[0]") << ";" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;

  return stream.str();
}

reduce_1d::reduce_1d(reduce_1d::parameters_type const & parameters,
                                       binding_policy_t binding) : base_impl<reduce_1d, reduce_1d_parameters>(parameters, binding)
{ }

reduce_1d::reduce_1d(unsigned int simd, unsigned int ls, unsigned int ng,
                               fetching_policy_type fetch, binding_policy_t bind):
    base_impl<reduce_1d, reduce_1d_parameters>(reduce_1d_parameters(simd,ls,ng,fetch), bind)
{}

std::vector<int_t> reduce_1d::input_sizes(expression_tree const  & x) const
{
  std::vector<size_t> reduce_1ds_idx = filter(x, &is_reduce_1d);
  int_t N = vector_size(lhs_most(x.tree(), reduce_1ds_idx[0]));
  return {N};
}

void reduce_1d::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, base & fallback, execution_handler const & control)
{
  expression_tree const  & x = control.x();

  //Preprocessing
  int_t size = input_sizes(x)[0];

  //fallback
  if(p_.simd_width > 1 && (requires_fallback(x) || (size%p_.simd_width>0)))
  {
      fallback.enqueue(queue, program, "fallback", fallback, control);
      return;
  }

  std::vector<expression_tree::node const *> reduce_1ds;
    std::vector<size_t> reduce_1ds_idx = filter(x, &is_reduce_1d);
    for (size_t idx: reduce_1ds_idx)
      reduce_1ds.push_back(&x.tree()[idx]);

  //Kernel
  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;

  driver::Kernel kernels[2] = { driver::Kernel(program,name[0].c_str()), driver::Kernel(program,name[1].c_str()) };

  //NDRange
  driver::NDRange global[2] = { driver::NDRange(p_.local_size_0*p_.num_groups), driver::NDRange(p_.local_size_0) };
  driver::NDRange local[2] = { driver::NDRange(p_.local_size_0), driver::NDRange(p_.local_size_0) };
  //Arguments
  for (auto & kernel : kernels)
  {
    unsigned int n_arg = 0;
    kernel.setSizeArg(n_arg++, size);
    kernel.setArg(n_arg++, driver::backend::workspaces::get(queue));
    set_arguments(x, kernel, n_arg, binding_policy_);
  }

  for (unsigned int k = 0; k < 2; k++)
    control.execution_options().enqueue(program.context(), kernels[k], global[k], local[k]);
  queue.synchronize();
}

}
}
