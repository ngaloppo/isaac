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


#include "isaac/symbolic/engine/process.h"
#include "isaac/templates/engine/keywords.h"
#include "isaac/templates/engine/stream.h"
#include "isaac/templates/reduce_2d.h"

#include "tools/arguments.hpp"
#include "tools/loop.hpp"
#include "tools/reductions.hpp"
#include "tools/vector_types.hpp"

#include <string>

namespace isaac
{
namespace templates
{

reduce_2d_parameters::reduce_2d_parameters(unsigned int _simd_width,
                              unsigned int _local_size_0, unsigned int _local_size_1,
                              unsigned int _num_groups_0, unsigned int _num_groups_1, fetching_policy_type _fetch_policy): base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1),
num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetch_policy(_fetch_policy) { }


int reduce_2d::is_invalid_impl(driver::Device const &, expression_tree const &) const
{
  if (p_.fetch_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

unsigned int reduce_2d::lmem_usage(const expression_tree&) const
{
  return (p_.local_size_0+1)*p_.local_size_1;
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
  driver::backend_type backend = device.backend();
  kernel_generation_stream stream(backend);

  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;

  auto unroll_tmp = [&]()
  {
      unsigned int offset = 0;
      for (symbolic::reduce_2d* rd : reductions)
      {
        numeric_type dtype = lhs_most(tree.data(),  tree.root()).lhs.dtype;
        std::string sdtype = to_string(dtype);
        if (is_index_reduction(rd->op()))
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

  int col_simd_width = (reduce_1d_type_ == REDUCE_COLUMNS) ? 1 : p_.simd_width;
  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"helper_math.h\"" << std::endl; break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl; break;
  }

  stream << "$KERNEL void " << name[0] << "($SIZE_T M, $SIZE_T N, $GLOBAL char* tmp, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  unroll_tmp();

  unsigned int local_size_0_ld = p_.local_size_0;
  std::string local_size_0_ld_str = to_string(local_size_0_ld);

  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("$LOCAL " + append_width("#scalartype", col_simd_width) + " #name_buf[" + to_string(p_.local_size_1*local_size_0_ld) + "];") << std::endl;

  stream << "for($SIZE_T r = $GLOBAL_IDX_1*" << col_simd_width << "; r < (M +" << p_.local_size_1 - 1 << ")/" << p_.local_size_1 << "*" << p_.local_size_1*col_simd_width << "; r += $GLOBAL_SIZE_1*" << col_simd_width << ")" << std::endl;
  stream << "{" << std::endl;

  stream.inc_tab();
  stream << "$SIZE_T lidx = $LOCAL_IDX_0;" << std::endl;
  stream << "$SIZE_T lidy = $LOCAL_IDX_1;" << std::endl;

  for (symbolic::reduce_2d* rd : reductions){
    std::string data_type = append_width("#scalartype",col_simd_width);

    stream << rd->process(data_type + " #name_acc = " + InitPrefix(backend, data_type).get()  + "(" + neutral_element((rd)->op(), backend, "#scalartype") + ");") << std::endl;
  }

  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  element_wise_loop_1D(stream, p_.fetch_policy, (reduce_1d_type_==REDUCE_COLUMNS)?p_.simd_width:1, "c", "N", "$GLOBAL_IDX_0", "$GLOBAL_SIZE_0", device, [&](unsigned int row_simd_width)
  {
    std::string rdtype = append_width("#scalartype", row_simd_width);
    std::string cdtype = append_width("#scalartype", col_simd_width);

    std::set<std::string> fetched;
    for (symbolic::reduce_2d* rd : reductions)
      for(symbolic::array* sym: symbolic::extract<symbolic::array>(tree, symbols, rd->index(), PARENT_NODE_TYPE)){
        if(fetched.insert(sym->process("#name")).second){
          if(reduce_1d_type_==REDUCE_COLUMNS)
            stream << sym->process(rdtype + " #name = " + vload(row_simd_width, "#scalartype", "c*#stride", "#pointer + r*#ld", "1", backend,false)+";") << std::endl;
          else
            stream << sym->process(cdtype + " #name = " + vload(col_simd_width, "#scalartype", "0", "#pointer + r*#stride + c*#ld", "1", backend,false) + ";") << std::endl;
        }
      }

    for (symbolic::reduce_2d* rd : reductions)
      for (unsigned int s = 0; s < row_simd_width; ++s)
      {
        std::string value = symbols.at({rd->index(), LHS_NODE_TYPE})->evaluate(access_vector_type("#name", s, row_simd_width));
        if (is_index_reduction(rd->op()))
          compute_index_reduce_1d(stream, rd->process("#name_acc"), "c*"+to_string(row_simd_width) + to_string(s), rd->process("#name_acc_value"), value, rd->op());
        else
          compute_reduce_1d(stream, rd->process("#name_acc"), value,rd->op());
      }
  });
  stream.dec_tab();
  stream << "}" << std::endl;

  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("#name_buf[lidy*" + local_size_0_ld_str + "+ lidx] = #name_acc;") << std::endl;

  stream << "#pragma unroll" << std::endl;
  stream << "for($SIZE_T stride = " << p_.local_size_0/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "$LOCAL_BARRIER;" << std::endl;
  stream <<  "if (lidx < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (symbolic::reduce_2d* rd : reductions)
    if (is_index_reduction(rd->op()))
      compute_index_reduce_1d(stream, rd->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx]"), rd->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx + stride]")
                                    , rd->process("#name_buf_value[lidy*" + local_size_0_ld_str + " + lidx]"), rd->process("#name_buf_value[lidy*" + local_size_0_ld_str + " + lidx + stride]")
                                    , rd->op());
    else
      compute_reduce_1d(stream,rd->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx]"), rd->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx + stride]"), rd->op());

  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;


  stream <<  "if (lidx == 0 && r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  if(p_.num_groups_0==1)
  {
//    std::map<std::string, std::string> accessors;
//    for(int s = 0 ; s < col_simd_width ; ++s)
//    {
//        accessors["reduce_2d"] = "#name_buf[lidy*" + local_size_0_ld_str + "]";
//        if(col_simd_width > 1)
//            accessors["reduce_2d"] = access_vector_type(accessors["reduce_2d"], s);
//        accessors["arrayn"] = "#pointer[(r +" + to_string(s) + ")*#stride]";
//        accessors["array1n"] = "#pointer[(r +" + to_string(s) + ")*#stride]";
//        accessors["arrayn1"] = "#pointer[(r +" + to_string(s) + ")*#stride]";
//        stream << evaluate(PARENT_NODE_TYPE, accessors, expression, expression.root(), mapping) << ";" << std::endl;
//    }
  }
  else
  {
    for (symbolic::reduction const * rd : reductions)
    {
      if(col_simd_width > 1)
          stream << "if(M - r > " << col_simd_width << "){" << std::endl;
      if (is_index_reduction(rd->op()))
          stream << rd->process(vstore(col_simd_width,"uint", "#name_buf_value[lidy*" + local_size_0_ld_str + "]", "0", "#name_temp_value + r + M*$GROUP_IDX_0", "1", backend, false)) << ";" << std::endl;
      stream << rd->process(vstore(col_simd_width,"#scalartype", "#name_buf[lidy*" + local_size_0_ld_str + "]", "0", "#name_temp + r + M*$GROUP_IDX_0", "1", backend, false)) << ";" << std::endl;
      if(col_simd_width > 1)
      {
          stream << "}" << std::endl;
          stream << "else{" << std::endl;
          stream.inc_tab();
          for(int s = 0 ; s < col_simd_width ; ++s){
              if (is_index_reduction(rd->op()))
                  stream << "if(r + " << s << "< M) " << rd->process("#name_temp_value[r + " + to_string(s) + " + M*$GROUP_IDX_0] = " + access_vector_type("#name_buf_value[lidy*" + local_size_0_ld_str + "]", s)) << ";" << std::endl;
              stream << "if(r + " << s << "< M) " << rd->process("#name_temp[r + " + to_string(s) + " + M*$GROUP_IDX_0] = " + access_vector_type("#name_buf[lidy*" + local_size_0_ld_str + "]", s)) << ";" << std::endl;
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


  if(p_.num_groups_0>1)
  {
  /////////////////////////////////////////
  ////////////// Kernel 2
  ////////////////////////////////////////

  if(backend==driver::OPENCL)
    stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl;

  stream << "$KERNEL void " << name[1] << "($SIZE_T M, $SIZE_T N , $GLOBAL char* tmp, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  unroll_tmp();

  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("$LOCAL #scalartype #name_buf[" + to_string(p_.local_size_1*local_size_0_ld) + "];") << std::endl;

  stream << "for($SIZE_T r = $GLOBAL_IDX_1; r < (M +" << p_.local_size_1 - 1 << ")/" << p_.local_size_1 << "*" << p_.local_size_1 << "; r += " << GlobalSize1(backend) << "){" << std::endl;
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
    stream << rd->process("#name_buf[lidy*" + local_size_0_ld_str + "+ lidx] = #name_acc;") << std::endl;

  stream << "#pragma unroll" << std::endl;
  stream << "for($SIZE_T stride = " << p_.local_size_0/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "$LOCAL_BARRIER;" << std::endl;
  stream <<  "if (lidx < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (symbolic::reduce_2d* rd : reductions)
    if (is_index_reduction(rd->op()))
      compute_index_reduce_1d(stream, rd->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx]"), rd->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx + stride]")
                                    , rd->process("#name_buf_value[lidy*" + local_size_0_ld_str + " + lidx]"), rd->process("#name_buf_value[lidy*" + local_size_0_ld_str + " + lidx + stride]")
                                    , rd->op());
    else
      compute_reduce_1d(stream,rd->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx]"), rd->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx + stride]"), rd->op());

  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;


  stream <<  "if (lidx == 0 && r < M)";
  stream << "{" << std::endl;
  stream.inc_tab();

//  std::map<std::string, std::string> accessors;
//  accessors["reduce_2d"] = "#name_buf[lidy*" + local_size_0_ld_str + "]";
//  accessors["arrayn"] = "#pointer[r*#stride]";
//  accessors["array1n"] = "#pointer[r*#stride]";
//  accessors["arrayn1"] = "#pointer[r*#stride]";
//  stream << evaluate(PARENT_NODE_TYPE, accessors, expression, expression.root(), mapping) << ";" << std::endl;

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
                                         reduce_2d::reduce_1d_type rtype,
                                         fusion_policy_t fusion_policy) :
  base_impl<reduce_2d, reduce_2d_parameters>(parameters, fusion_policy),
  reduce_1d_type_(rtype){ }

std::vector<int_t> reduce_2d::input_sizes(expression_tree const & expression) const
{
  std::vector<std::size_t> idx = symbolic::filter(expression, &is_reduce_1d);
  std::pair<int_t, int_t> MN = matrix_size(expression.data(), lhs_most(expression.data(), idx[0]));
  if(reduce_1d_type_==REDUCE_COLUMNS)
    std::swap(MN.first,MN.second);
  return {MN.first, MN.second};
}

void reduce_2d::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, base & fallback, execution_handler const & control)
{
  expression_tree const & expression = control.x();

  std::vector<int_t> MN = input_sizes(expression);
  std::vector<expression_tree::node const *> reduce_1ds;
  std::vector<size_t> reduce_1ds_idx = symbolic::filter(expression, &is_reduce_1d);
  for (size_t idx : reduce_1ds_idx)
    reduce_1ds.push_back(&expression.data()[idx]);

  //Fallback
  if(p_.simd_width>1 && requires_fallback(expression))
  {
      fallback.enqueue(queue, program, "fallback", fallback, control);
      return;
  }

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
    symbolic::set_arguments(expression, kernel, n_arg, fusion_policy_);
  }

  //NDRange
  driver::NDRange global[2] = { driver::NDRange(p_.local_size_0*p_.num_groups_0, p_.local_size_1*p_.num_groups_1), driver::NDRange(p_.local_size_0, p_.local_size_1*p_.num_groups_1) };
  driver::NDRange local[2] = { driver::NDRange(p_.local_size_0, p_.local_size_1), driver::NDRange(p_.local_size_0, p_.local_size_1) };
  for(unsigned int i = 0 ; i < nk ; ++i)
    control.execution_options().enqueue(program.context(), kernels[i], global[i], local[i]);
}

reduce_2d_rows::reduce_2d_rows(reduce_2d_parameters  const & parameters,fusion_policy_t fusion_policy): reduce_2d(parameters, REDUCE_ROWS, fusion_policy){}

reduce_2d_rows::reduce_2d_rows(unsigned int simd, unsigned int ls1, unsigned int ls2,  unsigned int ng1, unsigned int ng2,
               fetching_policy_type fetch, fusion_policy_t bind): reduce_2d(reduce_2d_parameters(simd, ls1, ls2, ng1, ng2, fetch), REDUCE_ROWS, bind) {}

reduce_2d_cols::reduce_2d_cols(reduce_2d::parameters_type  const & parameters, fusion_policy_t fusion_policy): reduce_2d(parameters, REDUCE_COLUMNS, fusion_policy){}

reduce_2d_cols::reduce_2d_cols(unsigned int simd, unsigned int ls1, unsigned int ls2, unsigned int ng1, unsigned int ng2,
               fetching_policy_type fetch, fusion_policy_t bind): reduce_2d(reduce_2d_parameters(simd, ls1, ls2, ng1, ng2, fetch), REDUCE_COLUMNS, bind) {}


}
}
