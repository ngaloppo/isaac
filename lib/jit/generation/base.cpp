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

#include <cassert>
#include <algorithm>
#include <string>

#include "isaac/array.h"
#include "isaac/jit/generation/base.h"
#include "isaac/exception/api.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{
namespace templates
{

/* stream */
base::genstream::buf::buf(std::ostringstream& oss,size_t const & tab_count) :
  oss_(oss), tab_count_(tab_count)
{ }


int base::genstream::buf::sync()
{
  for (size_t i=0; i<tab_count_;++i)
    oss_ << "  ";
  std::string next = str();
  oss_ << next;
  str("");
  return !oss_;
}

base::genstream::buf:: ~buf()
{  pubsync(); }

void base::genstream::process(std::string& str)
{

#define ADD_KEYWORD(NAME, OPENCL_NAME, CUDA_NAME) tools::find_and_replace(str, "$" + std::string(NAME), (backend_==driver::CUDA)?CUDA_NAME:OPENCL_NAME);


ADD_KEYWORD("GLOBAL_IDX_0", "get_global_id(0)", "(blockIdx.x*blockDim.x + threadIdx.x)")
ADD_KEYWORD("GLOBAL_IDX_1", "get_global_id(1)", "(blockIdx.y*blockDim.y + threadIdx.y)")
ADD_KEYWORD("GLOBAL_IDX_2", "get_global_id(2)", "(blockIdx.z*blockDim.z + threadIdx.z)")

ADD_KEYWORD("GLOBAL_SIZE_0", "get_global_size(0)", "(blockDim.x*gridDim.x)")
ADD_KEYWORD("GLOBAL_SIZE_1", "get_global_size(1)", "(blockDim.y*gridDim.y)")
ADD_KEYWORD("GLOBAL_SIZE_2", "get_global_size(2)", "(blockDim.z*gridDim.z)")

ADD_KEYWORD("LOCAL_IDX_0", "get_local_id(0)", "threadIdx.x")
ADD_KEYWORD("LOCAL_IDX_1", "get_local_id(1)", "threadIdx.y")
ADD_KEYWORD("LOCAL_IDX_2", "get_local_id(2)", "threadIdx.z")

ADD_KEYWORD("LOCAL_SIZE_0", "get_local_size(0)", "blockDim.x")
ADD_KEYWORD("LOCAL_SIZE_1", "get_local_size(1)", "blockDim.y")
ADD_KEYWORD("LOCAL_SIZE_2", "get_local_size(2)", "blockDim.z")

ADD_KEYWORD("GROUP_IDX_0", "get_group_id(0)", "blockIdx.x")
ADD_KEYWORD("GROUP_IDX_1", "get_group_id(1)", "blockIdx.y")
ADD_KEYWORD("GROUP_IDX_2", "get_group_id(2)", "blockIdx.z")

ADD_KEYWORD("GROUP_SIZE_0", "get_num_groups(0)", "GridDim.x")
ADD_KEYWORD("GROUP_SIZE_1", "get_num_groups(1)", "GridDim.y")
ADD_KEYWORD("GROUP_SIZE_2", "get_num_groups(2)", "GridDim.z")

ADD_KEYWORD("LOCAL_BARRIER", "barrier(CLK_LOCAL_MEM_FENCE)", "__syncthreads()")
ADD_KEYWORD("LOCAL_PTR", "__local", "")

ADD_KEYWORD("LOCAL", "__local", "__shared__")
ADD_KEYWORD("GLOBAL", "__global", "")

ADD_KEYWORD("SIZE_T", "int", "int")
ADD_KEYWORD("KERNEL", "__kernel", "extern \"C\" __global__")

ADD_KEYWORD("MAD", "mad", "fma")

#undef ADD_KEYWORD
}

base::genstream::genstream(driver::backend_type backend) : std::ostream(new buf(oss,tab_count_)), tab_count_(0), backend_(backend)
{

}

base::genstream::~genstream()
{
  delete rdbuf();
}

std::string base::genstream::str()
{
  std::string next = oss.str();
  process(next);
  return next;
}

void base::genstream::inc_tab()
{
  ++tab_count_;
}

void base::genstream::dec_tab()
{
  --tab_count_;
}

/* base */
base::base(size_t s, size_t ls0, size_t ls1):
  simd_width(s), local_size_0(ls0), local_size_1(ls1)
{}

size_t base::lmem_usage(expression_tree const  &) const
{
  return 0;
}

size_t base::registers_usage(expression_tree const  &) const
{
  return 0;
}

size_t base::temporary_workspace(expression_tree const  &) const
{
  return 0;
}

int base::is_invalid_impl(driver::Device const &, expression_tree const  &) const
{
  return TEMPLATE_VALID;
}

int base::is_invalid(expression_tree const  & tree, driver::Device const & device) const
{
  //Query device informations
  size_t lmem_available = device.local_mem_size();
  size_t lmem_used = lmem_usage(tree);
  if (lmem_used>lmem_available)
    return TEMPLATE_LOCAL_MEMORY_OVERFLOW;

  //Invalid work group size
  size_t max_workgroup_size = device.max_work_group_size();
  std::vector<size_t> max_work_item_sizes = device.max_work_item_sizes();
  if (local_size_0*local_size_1 > max_workgroup_size)
    return TEMPLATE_WORK_GROUP_SIZE_OVERFLOW;
  if (local_size_0 > max_work_item_sizes[0])
    return TEMPLATE_LOCAL_SIZE_0_OVERFLOW;

  if (local_size_1 > max_work_item_sizes[1])
    return TEMPLATE_LOCAL_SIZE_1_OVERFLOW;

  //Invalid SIMD Width
  if (simd_width!=1 && simd_width!=2 && simd_width!=3 && simd_width!=4)
    return TEMPLATE_INVALID_SIMD_WIDTH;

  return is_invalid_impl(device, tree);
}

std::string base::generate(std::string const & suffix, expression_tree const  & expression, driver::Device const & device)
{
  int err = is_invalid(expression, device);
  if(err != 0)
    throw operation_not_supported_exception("The supplied parameters for this template are invalid : err " + tools::to_string(err));

  //Create mapping
  symbolic::symbols_table mapping = symbolic::symbolize(expression);
  return generate_impl(suffix, expression, device, mapping);
}

}
}
