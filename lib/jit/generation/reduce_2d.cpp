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

#include "isaac/driver/kernel.h"
#include "isaac/driver/ndrange.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/jit/generation/reduce_2d.h"
#include "tools/arguments.hpp"
#include "tools/vector_types.hpp"

#include <string>

namespace isaac
{
namespace templates
{

int reduce_2d::is_invalid_impl(driver::Device const &, expression_tree const &) const
{
  if (fetch_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

size_t reduce_2d::lmem_usage(const expression_tree&) const
{
  return (local_size_0+1)*local_size_1;
}

size_t reduce_2d::temporary_workspace(expression_tree const & expressions) const
{
    std::vector<int_t> MN = input_sizes(expressions);
    int_t M = MN[0];
    if(num_groups_0 > 1)
      return M*num_groups_0;
    return 0;
}

std::string reduce_2d::generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  using tools::to_string;


  std::vector<symbolic::reduce_2d*> reductions = symbolic::extract<symbolic::reduce_2d>(tree, symbols);
  std::vector<std::size_t> assignments = symbolic::assignments(tree);
  driver::backend_type backend = device.backend();
  genstream stream(backend);

  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;

  int col_simd_width = (reduction_type_ == REDUCE_COLUMNS) ? 1 : simd_width;
  size_t local_size_0_ld = local_size_0;
  std::string ls0ldstr = to_string(local_size_0_ld);

  auto unroll_tmp = [&]()
  {
      size_t offset = 0;
      for (symbolic::reduce_2d* rd : reductions)
      {
        numeric_type dtype = tree.dtype();
        std::string sdtype = to_string(dtype);
        if (is_indexing(rd->op().type))
        {
          stream << rd->process("$GLOBAL uint* #name_temp = ($GLOBAL uint*)(tmp + " + tools::to_string(offset) + "*M);");
          offset += 4*num_groups_0;
          stream << rd->process("$GLOBAL " + sdtype + "* #name_temp_value = ($GLOBAL " + sdtype + "*)(tmp + " + tools::to_string(offset) + "*M);");
          offset += size_of(dtype)*num_groups_0;
        }
        else{
          stream << rd->process("$GLOBAL " + sdtype + "* #name_temp = ($GLOBAL " + sdtype + "*)(tmp + " + tools::to_string(offset) + "*M);");
          offset += size_of(dtype)*num_groups_0;
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
      stream << " __attribute__((reqd_work_group_size(" << local_size_0 << "," << local_size_1 << ",1)))" << std::endl;
      break;
  }
  stream << "$KERNEL void " << name[0] << "($SIZE_T M, $SIZE_T N, $GLOBAL char* tmp, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  //Unroll
  unroll_tmp();
  //Declare Buffers
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("$LOCAL " + append_width("#scalartype", col_simd_width) + " #name_buf[" + to_string(local_size_1*local_size_0_ld) + "];") << std::endl;
  //Loop r
  stream << "for($SIZE_T r = $GLOBAL_IDX_1*" << col_simd_width << "; r < (M +" << local_size_1 - 1 << ")/" << local_size_1 << "*" << local_size_1*col_simd_width << "; r += $GLOBAL_SIZE_1*" << col_simd_width << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << "$SIZE_T lidx = $LOCAL_IDX_0;" << std::endl;
  stream << "$SIZE_T lidy = $LOCAL_IDX_1;" << std::endl;
  //Accumulators
  for (symbolic::reduce_2d* rd : reductions){
    std::string data_type = append_width("#scalartype",col_simd_width);
    std::string make_prefix = (backend==driver::CUDA)?"make_"+data_type:"";
    stream << rd->process(data_type + " #name_acc = " + make_prefix  + "(" + neutral_element((rd)->op(), backend, "#scalartype") + ");") << std::endl;
  }
  //Loop c
  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  for_loop(stream, fetch_policy, (reduction_type_==REDUCE_COLUMNS)?simd_width:1, "c", "N", "$GLOBAL_IDX_0", "$GLOBAL_SIZE_0", [&](size_t row_simd_width)
  {
    std::string rdtype = append_width("#scalartype", row_simd_width);
    std::string cdtype = append_width("#scalartype", col_simd_width);
    //Fetch
    std::set<std::string> fetched;
    for (symbolic::reduce_2d* rd : reductions)
      for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, rd->root(), false))
        if(fetched.insert(sym->process("#name")).second){
          if(reduction_type_==REDUCE_COLUMNS)
            stream << sym->process(rdtype + " #name = " + append_width("loadv",row_simd_width) + "(c,r);") << std::endl;
          else
            stream << sym->process(cdtype + " #name = " + append_width("loadv",col_simd_width) + "(r,c);") << std::endl;
        }
    //Compute
    for (symbolic::reduce_2d* rd : reductions)
      for (size_t s = 0; s < row_simd_width; ++s){
        std::string value = rd->lhs()->evaluate({{"leaf", access_vector_type("#name", s, row_simd_width)}});
        if (is_indexing(rd->op().type))
          compute_index_reduce_1d(stream, rd->process("#name_acc"), "c*"+to_string(row_simd_width) + to_string(s), rd->process("#name_acc_value"), value, rd->op());
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
  stream << "for($SIZE_T stride = " << local_size_0/2 << "; stride >0; stride /=2)" << std::endl;
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
  stream <<  "if (lidx == 0 && r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  if(num_groups_0==1)
    for(size_t idx: assignments)
      for(int s = 0 ; s < col_simd_width ; ++s)
          stream << symbols.at(idx)->evaluate({{"leaf", "at(r+" + to_string(s) + ")"},
                                                {"reduce_2d", access_vector_type("#name_buf[lidy*" + ls0ldstr + "]", s, col_simd_width)}}) << ";" << std::endl;
  else
  {
    for (symbolic::reduction const * rd : reductions)
    {
      if(col_simd_width > 1)
          stream << "if(M - r > " << col_simd_width << "){" << std::endl;
      if (is_indexing(rd->op().type))
          stream << rd->process(vstore(col_simd_width,"uint", "#name_buf_value[lidy*" + ls0ldstr + "]", "0", "#name_temp_value + r + M*$GROUP_IDX_0", "1", backend, false)) << ";" << std::endl;
      stream << rd->process(vstore(col_simd_width,"#scalartype", "#name_buf[lidy*" + ls0ldstr + "]", "0", "#name_temp + r + M*$GROUP_IDX_0", "1", backend, false)) << ";" << std::endl;
      if(col_simd_width > 1)
      {
          stream << "}" << std::endl;
          stream << "else{" << std::endl;
          stream.inc_tab();
          for(int s = 0 ; s < col_simd_width ; ++s){
              if (is_indexing(rd->op().type))
                  stream << "if(r + " << s << "< M) " << rd->process("#name_temp_value[r + " + to_string(s) + " + M*$GROUP_IDX_0] = " + access_vector_type("#name_buf_value[lidy*" + ls0ldstr + "]", s)) << ";" << std::endl;
              stream << "if(r + " << s << "< M) " << rd->process("#name_temp[r + " + to_string(s) + " + M*$GROUP_IDX_0] = " + access_vector_type("#name_buf[lidy*" + ls0ldstr + "]", s)) << ";" << std::endl;
          }
          stream.dec_tab();
          stream << "}" << std::endl;
      }
    }
  }
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;


  /* ------------------------
   * Kernel 2
   * -----------------------*/
  if(num_groups_0>1)
  {
  if(backend==driver::OPENCL)
    stream << " __attribute__((reqd_work_group_size(" << local_size_0 << "," << local_size_1 << ",1)))" << std::endl;
  stream << "$KERNEL void " << name[1] << "($SIZE_T M, $SIZE_T N , $GLOBAL char* tmp, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  unroll_tmp();
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("$LOCAL #scalartype #name_buf[" + to_string(local_size_1*local_size_0_ld) + "];") << std::endl;
  stream << "for($SIZE_T r = $GLOBAL_IDX_1; r < (M +" << local_size_1 - 1 << ")/" << local_size_1 << "*" << local_size_1 << "; r += $GLOBAL_SIZE_1){" << std::endl;
  stream.inc_tab();
  stream << "$SIZE_T lidx = $LOCAL_IDX_0;" << std::endl;
  stream << "$SIZE_T lidy = $LOCAL_IDX_1;" << std::endl;
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("#scalartype #name_acc = " + neutral_element((rd)->op(), backend, "#scalartype") + ";") << std::endl;
  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << "for($SIZE_T c = lidx; c < " << num_groups_0 << "; c += $LOCAL_SIZE_0){" << std::endl;
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
  stream << "for($SIZE_T stride = " << local_size_0/2 << "; stride >0; stride /=2)" << std::endl;
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

reduce_2d::reduce_2d(size_t simd, size_t ls0, size_t ls1, size_t ng0, size_t ng1, fetching_policy_type fetch, token_family type):
  base(simd, ls0, ls1), num_groups_0(ng0), num_groups_1(ng1), fetch_policy(fetch), reduction_type_(type)
{ }

std::vector<int_t> reduce_2d::input_sizes(expression_tree const & tree) const
{
  std::vector<size_t> idx = symbolic::find(tree, [this](expression_tree::node const & x){return x.binary_operator.op.family==reduction_type_;});
  std::vector<int_t> shape = tree[tree[idx[0]].binary_operator.lhs].shape;
  if(reduction_type_==REDUCE_COLUMNS)
    return {shape[1], shape[0]};
  return {shape[0], shape[1]};
}

void reduce_2d::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix,
                        expression_tree const & tree, runtime::environment const & opt)
{
  std::vector<int_t> MN = input_sizes(tree);

  //Kernel
  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;
  size_t nk = (num_groups_0==1)?1:2;
  std::vector<driver::Kernel> kernels;
  for(size_t k = 0 ; k < nk ; ++k)
    kernels.push_back(driver::Kernel(program, name[k].c_str()));
  for(size_t k = 0 ; k < nk ; ++k)
  {
    driver::Kernel & kernel = kernels[k];
    unsigned int n_arg = 0;
    int_t M = MN[0];
    int_t N = MN[1];
    kernel.setSizeArg(n_arg++, M);
    kernel.setSizeArg(n_arg++, N);
    kernel.setArg(n_arg++, driver::backend::workspaces::get(queue)); //Temporary buffers
    symbolic::set_arguments(tree, kernel, n_arg);
  }

  //NDRange
  driver::NDRange global[2] = { driver::NDRange(local_size_0*num_groups_0, local_size_1*num_groups_1), driver::NDRange(local_size_0, local_size_1*num_groups_1) };
  driver::NDRange local[2] = { driver::NDRange(local_size_0, local_size_1), driver::NDRange(local_size_0, local_size_1) };
  for(size_t i = 0 ; i < nk ; ++i)
    opt.enqueue(queue.context(), kernels[i], global[i], local[i]);
}

reduce_2d_rows::reduce_2d_rows(size_t simd, size_t ls1, size_t ls2,  size_t ng1, size_t ng2,
               fetching_policy_type fetch): reduce_2d(simd, ls1, ls2, ng1, ng2, fetch, REDUCE_ROWS) {}

reduce_2d_cols::reduce_2d_cols(size_t simd, size_t ls1, size_t ls2, size_t ng1, size_t ng2,
               fetching_policy_type fetch): reduce_2d(simd, ls1, ls2, ng1, ng2, fetch, REDUCE_COLUMNS) {}


}
}
