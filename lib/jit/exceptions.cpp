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

#include "isaac/jit/exceptions.h"

namespace isaac
{

namespace jit
{

//
code_generation_error::code_generation_error(std::string message) :
  message_("Code generation error: " + message) {}

const char* code_generation_error::what() const throw()
{ return message_.c_str(); }

//
semantic_error::semantic_error(std::string const & str) :
  message_("Semantic error: " + str) {}

const char* semantic_error::what() const throw()
{ return message_.c_str(); }

}

}
