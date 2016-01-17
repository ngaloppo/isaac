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

#ifndef ISAAC_KERNELS_PARSE_FILTER_HPP
#define ISAAC_KERNELS_PARSE_FILTER_HPP

#include <cstring>
#include <vector>
#include "traverse.hpp"

namespace isaac
{

inline std::vector<size_t> filter(expression_tree const & expression, size_t idx, leaf_t leaf, std::function<bool (expression_tree::node const &)> const & pred)
{
  std::vector<size_t> result;
  auto fun = [&](size_t index, leaf_t leaf) {  if(leaf==PARENT_NODE_TYPE && pred(expression.data()[index])) result.push_back(index); };
  _traverse(expression, idx, leaf, fun);
  return result;
}

inline std::vector<size_t> filter(expression_tree const & expression, std::function<bool (expression_tree::node const &)> const & pred)
{ return filter(expression, expression.root(), PARENT_NODE_TYPE, pred); }


}

#endif
