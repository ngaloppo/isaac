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

#include <iostream>
#include <sstream>
#include <string>

#include "isaac/symbolic/expression/io.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{

#define ISAAC_MAP_TO_STRING(NAME) case NAME: return #NAME

inline std::string to_string(node_type const & f)
{
  switch(f)
  {
    ISAAC_MAP_TO_STRING(INVALID_SUBTYPE);
    ISAAC_MAP_TO_STRING(VALUE_SCALAR_TYPE);
    ISAAC_MAP_TO_STRING(DENSE_ARRAY_TYPE);
    default: return "UNKNOWN";
  }
}

inline std::string to_string(tree_node const & e)
{
  if(e.type==COMPOSITE_OPERATOR_TYPE)
  {
    return"COMPOSITE [" + tools::to_string(e.index) + "]";
  }
  return tools::to_string(e.type);
}

inline std::ostream & operator<<(std::ostream & os, expression_tree::node const & s_node)
{
  os << "LHS: " << to_string(s_node.lhs) << "|" << s_node.lhs.dtype << ", "
     << "OP: "  << s_node.op.type_family << " | " << s_node.op.type << ", "
     << "RHS: " << to_string(s_node.rhs) << "|" << s_node.rhs.dtype;

  return os;
}


namespace detail
{
  /** @brief Recursive worker routine for printing a whole expression_tree */
  inline void print_node(std::ostream & os, isaac::expression_tree const & s, size_t index, size_t indent = 0)
  {
    expression_tree::data_type const & nodes = s.data();
    expression_tree::node const & current_node = nodes[index];

    for (size_t i=0; i<indent; ++i)
      os << " ";

    os << "Node " << index << ": " << current_node << std::endl;

    if (current_node.lhs.type == COMPOSITE_OPERATOR_TYPE)
      print_node(os, s, current_node.lhs.index, indent+1);

    if (current_node.rhs.type == COMPOSITE_OPERATOR_TYPE)
      print_node(os, s, current_node.rhs.index, indent+1);
  }
}

std::string to_string(isaac::expression_tree const & s)
{
  std::ostringstream os;
  detail::print_node(os, s, s.root());
  return os.str();
}

}

