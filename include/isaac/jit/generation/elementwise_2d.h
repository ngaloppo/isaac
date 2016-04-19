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

#ifndef ISAAC_BACKEND_TEMPLATES_MAXPY_H
#define ISAAC_BACKEND_TEMPLATES_MAXPY_H

#include <vector>
#include "isaac/jit/generation/base.h"

namespace isaac
{
namespace templates
{

class elementwise_2d : public base
{
private:
  void check_valid_impl(driver::Device const &, expression const &) const;
  std::string generate_impl(std::string const & suffix, expression const  & expressions, driver::Device const & device, symbolic::symbols_table const & mapping) const;

public:
  elementwise_2d(size_t simd, size_t ls0, size_t ls1,  size_t ng0, size_t ng1, fetching_policy_type fetch);
  std::vector<int_t> input_sizes(expression const  & tree) const;
  void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, expression const & tree, runtime::environment const & opt);

private:
  size_t num_groups_0;
  size_t num_groups_1;
  fetching_policy_type fetching_policy;
};

}
}

#endif
