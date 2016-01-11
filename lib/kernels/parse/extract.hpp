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

#ifndef ISAAC_KERNELS_PARSE_EXTRACT_HPP
#define ISAAC_KERNELS_PARSE_EXTRACT_HPP

#include <cstring>

#include "traverse.hpp"
#include "isaac/kernels/parse.h"

namespace isaac
{

//Extract symbolic types
template<class T>
inline void extract(expression_tree const & expression, symbolic::mapping_type const & symbolic,
                    size_t idx, leaf_t leaf, std::set<std::string>& processed, std::vector<T*>& result)
{
  auto extract_impl = [&](size_t index, leaf_t leaf)
  {
    symbolic::mapping_type::const_iterator it = symbolic.find({index, leaf});
    if(it!=symbolic.end())
    {
      T* obj = dynamic_cast<T*>(&*it->second);
      if(obj && processed.insert(obj->process("#name")).second)
        result.push_back(obj);
    }
  };
  _traverse(expression, idx, leaf, extract_impl);
}

template<class T>
inline std::vector<T*> extract(expression_tree const & expression, symbolic::mapping_type const & symbolic, std::vector<size_t> idxs, leaf_t leaf = PARENT_NODE_TYPE)
{
  std::vector<T*> result;
  std::set<std::string> processed;
  for(size_t idx: idxs)
     extract(expression, symbolic, idx, leaf, processed, result);
  return result;
}

template<class T>
inline std::vector<T*> extract(expression_tree const & expression, symbolic::mapping_type const & symbolic, size_t root, leaf_t leaf = PARENT_NODE_TYPE)
{
  return extract<T>(expression, symbolic, std::vector<size_t>{root}, leaf);
}

template<class T>
inline std::vector<T*> extract(expression_tree const & expression, symbolic::mapping_type const & symbolic)
{
  return extract<T>(expression, symbolic, expression.root(), PARENT_NODE_TYPE);
}

}

#endif
