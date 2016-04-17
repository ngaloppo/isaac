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

base::base(unsigned int s, unsigned int ls0, unsigned int ls1):
  simd_width(s), local_size_0(ls0), local_size_1(ls1)
{}

unsigned int base::lmem_usage(expression_tree const  &) const
{
  return 0;
}

unsigned int base::registers_usage(expression_tree const  &) const
{
  return 0;
}

unsigned int base::temporary_workspace(expression_tree const  &) const
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
