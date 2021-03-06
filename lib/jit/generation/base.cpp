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
#include "isaac/jit/generation/engine/keywords.h"
#include "isaac/jit/generation/elementwise_1d.h"
#include "isaac/jit/generation/reduce_1d.h"
#include "isaac/jit/generation/elementwise_2d.h"
#include "isaac/jit/generation/reduce_2d.h"
#include "isaac/jit/generation/matrix_product.h"
#include "isaac/jit/generation/base.h"
#include "isaac/exception/api.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{
namespace templates
{

base::parameters_type::parameters_type(unsigned int _vwidth, int_t _ls0, int_t _ls1, int_t _num_kernels) : vwidth(_vwidth), ls0(_ls0), ls1(_ls1), num_kernels(_num_kernels)
{ }

base::base(fusion_policy_t fusion_policy) : fusion_policy_(fusion_policy)
{}

unsigned int base::lmem_usage(expression_tree const  &) const
{ return 0; }

unsigned int base::registers_usage(expression_tree const  &) const
{ return 0; }

unsigned int base::temporary_workspace(expression_tree const  &) const
{ return 0; }

base::~base()
{
}

std::string base::generate(std::string const & suffix, expression_tree const  & expression, driver::Device const & device)
{
  int err = is_invalid(expression, device);
  if(err != 0)
    throw operation_not_supported_exception("The supplied parameters for this template are invalid : err " + tools::to_string(err));

  //Create mapping
  symbolic::symbols_table mapping = symbolic::symbolize(fusion_policy_, expression);
  return generate_impl(suffix, expression, device, mapping);
}

template<class TType, class PType>
int base_impl<TType, PType>::is_invalid_impl(driver::Device const &, expression_tree const  &) const
{ return TEMPLATE_VALID; }

template<class TType, class PType>
base_impl<TType, PType>::base_impl(parameters_type const & parameters, fusion_policy_t fusion_policy) : base(fusion_policy), p_(parameters)
{ }

template<class TType, class PType>
unsigned int base_impl<TType, PType>::ls0() const
{ return p_.ls0; }

template<class TType, class PType>
unsigned int base_impl<TType, PType>::ls1() const
{ return p_.ls1; }

template<class TType, class PType>
std::shared_ptr<base> base_impl<TType, PType>::clone() const
{ return std::shared_ptr<base>(new TType(*dynamic_cast<TType const *>(this))); }

template<class TType, class PType>
int base_impl<TType, PType>::is_invalid(expression_tree const  & expressions, driver::Device const & device) const
{
  //Query device informations
  size_t lmem_available = device.local_mem_size();
  size_t lmem_used = lmem_usage(expressions);
  if (lmem_used>lmem_available)
    return TEMPLATE_LOCAL_MEMORY_OVERFLOW;

  //Invalid work group size
  size_t max_workgroup_size = device.max_work_group_size();
  std::vector<size_t> max_work_item_sizes = device.max_work_item_sizes();
  if (p_.ls0*p_.ls1 > max_workgroup_size)
    return TEMPLATE_WORK_GROUP_SIZE_OVERFLOW;
  if (p_.ls0 > max_work_item_sizes[0])
    return TEMPLATE_LOCAL_SIZE_0_OVERFLOW;

  if (p_.ls1 > max_work_item_sizes[1])
    return TEMPLATE_LOCAL_SIZE_1_OVERFLOW;

  //Invalid SIMD Width
  if (p_.vwidth!=1 && p_.vwidth!=2 && p_.vwidth!=3 && p_.vwidth!=4)
    return TEMPLATE_INVALID_SIMD_WIDTH;

  return is_invalid_impl(device, expressions);
}

template class base_impl<elementwise_1d, elementwise_1d_parameters>;
template class base_impl<reduce_1d, reduce_1d_parameters>;
template class base_impl<elementwise_2d, elementwise_2d_parameters>;
template class base_impl<reduce_2d, reduce_2d_parameters>;
template class base_impl<matrix_product, matrix_product_parameters>;

}
}
