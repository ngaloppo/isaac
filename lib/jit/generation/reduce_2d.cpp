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


#include "isaac/jit/syntax/engine/process.h"
#include "isaac/jit/generation/engine/keywords.h"
#include "isaac/jit/generation/engine/stream.h"
#include "isaac/jit/generation/reduce_2d.h"
#include "tools/arguments.hpp"
#include "tools/loop.hpp"
#include "tools/reductions.hpp"
#include "tools/vector_types.hpp"

#include <string>

namespace isaac
{
namespace templates
{

reduce_2d_parameters::reduce_2d_parameters(unsigned int _vwidth,
                              unsigned int _ls0, unsigned int _ls1,
                              unsigned int _num_groups_0, unsigned int _num_groups_1, fetch_type _fetch_policy): base::parameters_type(_vwidth, _ls0, _ls1, 1),
num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetch_policy(_fetch_policy) { }


int reduce_2d::is_invalid_impl(driver::Device const &, expression_tree const &) const
{
  if (p_.fetch_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

unsigned int reduce_2d::lmem_usage(const expression_tree&) const
{
  return (p_.ls0+1)*p_.ls1;
}

unsigned int reduce_2d::temporary_workspace(expression_tree const & expressions) const
{
    std::vector<int_t> MN = input_sizes(expressions);
    int_t M = MN[0];
    if(p_.num_groups_0 > 1)
      return M*p_.num_groups_0;
    return 0;
}

std::string reduce_2d::generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  using tools::to_string;


  std::vector<symbolic::reduce_2d*> reductions = symbolic::extract<symbolic::reduce_2d>(tree, symbols);
  std::vector<std::size_t> assignments = symbolic::assignments(tree);
  driver::backend_type backend = device.backend();
  kernel_generation_stream stream(backend);

  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;

  unsigned int ldls = p_.ls0;
  std::string ls0ldstr = to_string(ldls);

  auto unroll_tmp = [&]()
  {
      unsigned int offset = 0;
      for (symbolic::reduce_2d* rd : reductions)
      {
        numeric_type dtype = tree.dtype();
        std::string sdtype = to_string(dtype);
        if (is_indexing(rd->op().type))
        {
          stream << rd->process("$GLOBAL uint* #name_temp = ($GLOBAL uint*)(tmp + " + tools::to_string(offset) + "*M);");
          offset += 4*p_.num_groups_0;
          stream << rd->process("$GLOBAL " + sdtype + "* #name_temp_value = ($GLOBAL " + sdtype + "*)(tmp + " + tools::to_string(offset) + "*M);");
          offset += size_of(dtype)*p_.num_groups_0;
        }
        else{
          stream << rd->process("$GLOBAL " + sdtype + "* #name_temp = ($GLOBAL " + sdtype + "*)(tmp + " + tools::to_string(offset) + "*M);");
          offset += size_of(dtype)*p_.num_groups_0;
        }
      }
  };

  /* ------------------------
   * Kernel 1
   * -----------------------*/
  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"vector.h\"" << std::endl;
      break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << p_.ls0 << "," << p_.ls1 << ",1)))" << std::endl;
      break;
  }
  stream << "$KERNEL void " << name[0] << "($SIZE_T M, $SIZE_T N, $GLOBAL char* tmp, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  //Unroll
  unroll_tmp();
  stream << "$SIZE_T lidx = $LOCAL_IDX_0;" << std::endl;
  stream << "$SIZE_T lidy = $LOCAL_IDX_1;" << std::endl;
  //Loop r
  std::ostringstream upper;
  upper << "(M +" << p_.ls1 - 1 << ")/" << p_.ls1 << "*" << p_.ls1;

  element_wise_loop_1D(stream, p_.fetch_policy, (reduction_type_==REDUCE_ROWS)?1:1, "r", upper.str(), "$GLOBAL_IDX_1", "$GLOBAL_SIZE_1", device, [&](unsigned int cwidth)
  {
  //Declare Buffers
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("$LOCAL " + append_width("#scalartype", cwidth) + " #name_buf[" + to_string(p_.ls1*ldls) + "];") << std::endl;

  //Accumulators
  for (symbolic::reduce_2d* rd : reductions){
    std::string data_type = append_width("#scalartype",cwidth);
    stream << rd->process(data_type + " #name_acc = " + InitPrefix(backend, data_type).get()  + "(" + neutral_element((rd)->op(), backend, "#scalartype") + ");") << std::endl;
  }
  //Loop c
  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  element_wise_loop_1D(stream, p_.fetch_policy, (reduction_type_==REDUCE_COLUMNS)?p_.vwidth:1, "c", "N", "$GLOBAL_IDX_0", "$GLOBAL_SIZE_0", device, [&](unsigned int rwidth)
  {
    std::string rdtype = append_width("#scalartype", rwidth);
    std::string cdtype = append_width("#scalartype", cwidth);
    //Fetch
    std::set<std::string> fetched;
    for (symbolic::reduce_2d* rd : reductions)
      for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, rd->root(), false))
        if(fetched.insert(sym->process("#name")).second){
          if(reduction_type_==REDUCE_COLUMNS)
            stream << sym->process(rdtype + " #name = " + append_width("loadv",rwidth) + "(c,r);") << std::endl;
          else
            stream << sym->process(cdtype + " #name = " + append_width("loadv",cwidth) + "(r,c);") << std::endl;
        }
    //Compute
    for (symbolic::reduce_2d* rd : reductions)
      for (unsigned int s = 0; s < rwidth; ++s){
        std::string value = rd->lhs()->evaluate({{"leaf", access_vector_type("#name", s, rwidth)}});
        if (is_indexing(rd->op().type))
          compute_index_reduce_1d(stream, rd->process("#name_acc"), "c*"+to_string(rwidth) + to_string(s), rd->process("#name_acc_value"), value, rd->op());
        else
          compute_reduce_1d(stream, rd->process("#name_acc"), value,rd->op());
      }
  });
  stream.dec_tab();
  stream << "}" << std::endl;
  //Copy to local memory
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("#name_buf[lidy*" + ls0ldstr + "+ lidx] = #name_acc;") << std::endl;
  //Reduce local memory
  stream << "#pragma unroll" << std::endl;
  stream << "for($SIZE_T stride = " << p_.ls0/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << "$LOCAL_BARRIER;" << std::endl;
  stream <<  "if (lidx < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  for (symbolic::reduce_2d* rd : reductions)
    if (is_indexing(rd->op().type))
      compute_index_reduce_1d(stream, rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx + stride]")
                                    , rd->process("#name_buf_value[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf_value[lidy*" + ls0ldstr + " + lidx + stride]")
                                    , rd->op());
    else
      compute_reduce_1d(stream,rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx + stride]"), rd->op());
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  //Write result/temporary
  stream <<  "if (r < M && lidx == 0)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  if(p_.num_groups_0==1)
    for(size_t idx: assignments)
      for(size_t s = 0 ; s < cwidth ; ++s)
          stream << symbols.at(idx)->evaluate({{"leaf", "at(r+" + to_string(s) + ")"},
                                                {"reduce_2d", access_vector_type("#name_buf[lidy*" + ls0ldstr + "]", s, cwidth)}}) << ";" << std::endl;
  else
    for (symbolic::reduction const * rd : reductions)
      for(size_t s = 0 ; s < cwidth ; ++s){
          if (is_indexing(rd->op().type))
              stream << "if(r + " << s << "< M) " << rd->process("#name_temp_value[r + " + to_string(s) + " + M*$GROUP_IDX_0] = " + access_vector_type("#name_buf_value[lidy*" + ls0ldstr + "]", s, cwidth)) << ";" << std::endl;
          stream << "if(r + " << s << "< M) " << rd->process("#name_temp[r + " + to_string(s) + " + M*$GROUP_IDX_0] = " + access_vector_type("#name_buf[lidy*" + ls0ldstr + "]", s, cwidth)) << ";" << std::endl;
      }
  stream.dec_tab();
  stream << "}" << std::endl;
  });
  stream.dec_tab();
  stream << "}" << std::endl;


  /* ------------------------
   * Kernel 2
   * -----------------------*/
  if(p_.num_groups_0>1)
  {
  if(backend==driver::OPENCL)
    stream << " __attribute__((reqd_work_group_size(" << p_.ls0 << "," << p_.ls1 << ",1)))" << std::endl;
  stream << "$KERNEL void " << name[1] << "($SIZE_T M, $SIZE_T N , $GLOBAL char* tmp, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  unroll_tmp();
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("$LOCAL #scalartype #name_buf[" + to_string(p_.ls1*ldls) + "];") << std::endl;
  stream << "for($SIZE_T r = $GLOBAL_IDX_1; r < (M +" << p_.ls1 - 1 << ")/" << p_.ls1 << "*" << p_.ls1 << "; r += " << GlobalSize1(backend) << "){" << std::endl;
  stream.inc_tab();
  stream << "$SIZE_T lidx = $LOCAL_IDX_0;" << std::endl;
  stream << "$SIZE_T lidy = $LOCAL_IDX_1;" << std::endl;
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("#scalartype #name_acc = " + neutral_element((rd)->op(), backend, "#scalartype") + ";") << std::endl;
  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << "for($SIZE_T c = lidx; c < " << p_.num_groups_0 << "; c += $LOCAL_SIZE_0){" << std::endl;
  stream.inc_tab();
  for (symbolic::reduce_2d* rd: reductions)
    compute_reduce_1d(stream, rd->process("#name_acc"), rd->process("#name_temp[r + M*c]"), rd->op());
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("#name_buf[lidy*" + ls0ldstr + "+ lidx] = #name_acc;") << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "for($SIZE_T stride = " << p_.ls0/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << "$LOCAL_BARRIER;" << std::endl;
  stream <<  "if (lidx < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  for (symbolic::reduce_2d* rd : reductions)
    if (is_indexing(rd->op().type))
      compute_index_reduce_1d(stream, rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx + stride]")
                                    , rd->process("#name_buf_value[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf_value[lidy*" + ls0ldstr + " + lidx + stride]")
                                    , rd->op());
    else
      compute_reduce_1d(stream,rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx + stride]"), rd->op());
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  stream <<  "if (lidx == 0 && r < M)";
  stream << "{" << std::endl;
  stream.inc_tab();
  for(size_t idx: assignments)
    stream << symbols.at(idx)->evaluate({{"leaf", "at(r)"}, {"reduce_2d", "#name_buf[lidy*" + ls0ldstr + "]"}}) << ";" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  }

//  std::cout << stream.str() << std::endl;
  return stream.str();
}

reduce_2d::reduce_2d(reduce_2d::parameters_type const & parameters,
                                         operation_type_family rtype,
                                         fusion_policy_t fusion_policy) :
  base_impl<reduce_2d, reduce_2d_parameters>(parameters, fusion_policy),
  reduction_type_(rtype){ }

std::vector<int_t> reduce_2d::input_sizes(expression_tree const & tree) const
{
  std::vector<size_t> idx = symbolic::find(tree, [this](expression_tree::node const & x){return x.type==COMPOSITE_OPERATOR_TYPE && x.binary_operator.op.type_family==reduction_type_;});
  std::vector<int_t> shape = tree[tree[idx[0]].binary_operator.lhs].shape;
  if(reduction_type_==REDUCE_COLUMNS)
    return {shape[1], shape[0]};
  return {shape[0], shape[1]};
}

void reduce_2d::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const & control)
{
  expression_tree const & tree = control.x();
  std::vector<int_t> MN = input_sizes(tree);

  //Kernel
  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;

  unsigned int nk = (p_.num_groups_0==1)?1:2;

  std::vector<driver::Kernel> kernels;
  for(unsigned int k = 0 ; k < nk ; ++k)
    kernels.push_back(driver::Kernel(program, name[k].c_str()));

  for(unsigned int k = 0 ; k < nk ; ++k)
  {
    driver::Kernel & kernel = kernels[k];
    unsigned int n_arg = 0;
    int_t M = MN[0];
    int_t N = MN[1];
    kernel.setSizeArg(n_arg++, M);
    kernel.setSizeArg(n_arg++, N);
    kernel.setArg(n_arg++, driver::backend::workspaces::get(queue)); //Temporary buffers
    symbolic::set_arguments(tree, kernel, n_arg, fusion_policy_);
  }

  //NDRange
  driver::NDRange global[2] = { driver::NDRange(p_.ls0*p_.num_groups_0, p_.ls1*p_.num_groups_1), driver::NDRange(p_.ls0, p_.ls1*p_.num_groups_1) };
  driver::NDRange local[2] = { driver::NDRange(p_.ls0, p_.ls1), driver::NDRange(p_.ls0, p_.ls1) };
  for(unsigned int i = 0 ; i < nk ; ++i)
    control.execution_options().enqueue(program.context(), kernels[i], global[i], local[i]);
}

reduce_2d_rows::reduce_2d_rows(reduce_2d_parameters  const & parameters,fusion_policy_t fusion_policy): reduce_2d(parameters, REDUCE_ROWS, fusion_policy){}

reduce_2d_rows::reduce_2d_rows(unsigned int simd, unsigned int ls1, unsigned int ls2,  unsigned int ng1, unsigned int ng2,
               fetch_type fetch, fusion_policy_t bind): reduce_2d(reduce_2d_parameters(simd, ls1, ls2, ng1, ng2, fetch), REDUCE_ROWS, bind) {}

reduce_2d_cols::reduce_2d_cols(reduce_2d::parameters_type  const & parameters, fusion_policy_t fusion_policy): reduce_2d(parameters, REDUCE_COLUMNS, fusion_policy){}

reduce_2d_cols::reduce_2d_cols(unsigned int simd, unsigned int ls1, unsigned int ls2, unsigned int ng1, unsigned int ng2,
               fetch_type fetch, fusion_policy_t bind): reduce_2d(reduce_2d_parameters(simd, ls1, ls2, ng1, ng2, fetch), REDUCE_COLUMNS, bind) {}


}
}
