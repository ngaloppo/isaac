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

#include "isaac/common.h"
#include "isaac/exceptions.h"

namespace isaac
{

std::string to_string(numeric_type const & type)
{
  switch (type)
  {
  //  case BOOL_TYPE: return "bool";
    case CHAR_TYPE: return "char";
    case UCHAR_TYPE: return "uchar";
    case SHORT_TYPE: return "short";
    case USHORT_TYPE: return "ushort";
    case INT_TYPE:  return "int";
    case UINT_TYPE: return "uint";
    case LONG_TYPE:  return "long";
    case ULONG_TYPE: return "ulong";
  //  case HALF_TYPE : return "half";
    case FLOAT_TYPE : return "float";
    case DOUBLE_TYPE : return "double";
    default : throw unknown_datatype(type);
  }
}

numeric_type numeric_type_from_string(std::string const & name)
{
  if(name=="int8") return CHAR_TYPE;
  else if(name=="uint8") return UCHAR_TYPE;
  else if(name=="int16") return SHORT_TYPE;
  else if(name=="uint16") return USHORT_TYPE;
  else if(name=="int32") return INT_TYPE;
  else if(name=="uint32") return UINT_TYPE;
  else if(name=="int64") return LONG_TYPE;
  else if(name=="uint64") return ULONG_TYPE;
  else if(name=="float32") return FLOAT_TYPE;
  else if(name=="float64") return DOUBLE_TYPE;
  throw std::invalid_argument("Invalid datatype: " + name);
}

size_t size_of(numeric_type type)
{
  switch (type)
  {
//  case BOOL_TYPE:
  case UCHAR_TYPE:
  case CHAR_TYPE: return 1;

//  case HALF_TYPE:
  case USHORT_TYPE:
  case SHORT_TYPE: return 2;

  case UINT_TYPE:
  case INT_TYPE:
  case FLOAT_TYPE: return 4;

  case ULONG_TYPE:
  case LONG_TYPE:
  case DOUBLE_TYPE: return 8;

  default: throw unknown_datatype(type);
  }
}

}
