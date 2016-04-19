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

#ifndef ISAAC_VALUE_SCALAR_H
#define ISAAC_VALUE_SCALAR_H

#include "isaac/common.h"
#include <stdint.h>

namespace isaac
{

class device_scalar;
class expression;

union ISAACAPI values_holder
{
  int8_t int8;
  uint8_t uint8;
  int16_t int16;
  uint16_t uint16;
  int32_t int32;
  uint32_t uint32;
  int64_t int64;
  uint64_t uint64;
  float float32;
  double float64;
};

class ISAACAPI scalar
{
  template<class T> void init(T const &);
  template<class T> T cast() const;
public:
#define ISAAC_INSTANTIATE(TYPE) scalar(TYPE value, numeric_type dtype = to_numeric_type<TYPE>::value);
  ISAAC_INSTANTIATE(char)
  ISAAC_INSTANTIATE(unsigned char)
  ISAAC_INSTANTIATE(short)
  ISAAC_INSTANTIATE(unsigned short)
  ISAAC_INSTANTIATE(int)
  ISAAC_INSTANTIATE(unsigned int)
  ISAAC_INSTANTIATE(long)
  ISAAC_INSTANTIATE(unsigned long)
  ISAAC_INSTANTIATE(long long)
  ISAAC_INSTANTIATE(unsigned long long)
  ISAAC_INSTANTIATE(float)
  ISAAC_INSTANTIATE(double)
#undef ISAAC_INSTANTIATE
  scalar(values_holder values, numeric_type dtype);
  explicit scalar(device_scalar const &);
  explicit scalar(expression const &);
  explicit scalar(numeric_type dtype = INVALID_NUMERIC_TYPE);

  values_holder values() const;
  numeric_type dtype() const;

#define INSTANTIATE(type) operator type() const;
  INSTANTIATE(bool)
  INSTANTIATE(char)
  INSTANTIATE(unsigned char)
  INSTANTIATE(short)
  INSTANTIATE(unsigned short)
  INSTANTIATE(int)
  INSTANTIATE(unsigned int)
  INSTANTIATE(long)
  INSTANTIATE(unsigned long)
  INSTANTIATE(long long)
  INSTANTIATE(unsigned long long)
  INSTANTIATE(float)
  INSTANTIATE(double)
#undef INSTANTIATE
private:
  values_holder values_;
  numeric_type dtype_;
};

ISAACAPI scalar int8(int8_t v);
ISAACAPI scalar uint8(uint8_t v);
ISAACAPI scalar int16(int16_t v);
ISAACAPI scalar uint16(uint16_t v);
ISAACAPI scalar int32(int32_t v);
ISAACAPI scalar uint32(uint32_t v);
ISAACAPI scalar int64(int64_t v);
ISAACAPI scalar uint64(uint64_t v);
ISAACAPI scalar float32(float v);
ISAACAPI scalar float64(double v);

template<class T>
ISAACAPI T cast(isaac::scalar const &);

#define ISAAC_DECLARE_BINARY_OPERATOR(RET, OPNAME) \
ISAACAPI RET OPNAME (scalar const &, char  );\
ISAACAPI RET OPNAME (scalar const &, unsigned char );\
ISAACAPI RET OPNAME (scalar const &, short );\
ISAACAPI RET OPNAME (scalar const &, unsigned short);\
ISAACAPI RET OPNAME (scalar const &, int   );\
ISAACAPI RET OPNAME (scalar const &, unsigned int  );\
ISAACAPI RET OPNAME (scalar const &, long  );\
ISAACAPI RET OPNAME (scalar const &, unsigned long );\
ISAACAPI RET OPNAME (scalar const &, long long);\
ISAACAPI RET OPNAME (scalar const &, unsigned long long);\
ISAACAPI RET OPNAME (scalar const &, float );\
ISAACAPI RET OPNAME (scalar const &, double);\
ISAACAPI RET OPNAME (char   , scalar const &);\
ISAACAPI RET OPNAME (unsigned char  , scalar const &);\
ISAACAPI RET OPNAME (short  , scalar const &);\
ISAACAPI RET OPNAME (unsigned short , scalar const &);\
ISAACAPI RET OPNAME (int    , scalar const &);\
ISAACAPI RET OPNAME (unsigned int   , scalar const &);\
ISAACAPI RET OPNAME (long   , scalar const &);\
ISAACAPI RET OPNAME (unsigned long  , scalar const &);\
ISAACAPI RET OPNAME (long long, scalar const &);\
ISAACAPI RET OPNAME (unsigned long long, scalar const &);\
ISAACAPI RET OPNAME (float  , scalar const &);\
ISAACAPI RET OPNAME (double , scalar const &);\
ISAACAPI RET OPNAME (scalar const &, scalar const &);

ISAAC_DECLARE_BINARY_OPERATOR(scalar, operator +)
ISAAC_DECLARE_BINARY_OPERATOR(scalar, operator -)
ISAAC_DECLARE_BINARY_OPERATOR(scalar, operator *)
ISAAC_DECLARE_BINARY_OPERATOR(scalar, operator /)

ISAAC_DECLARE_BINARY_OPERATOR(scalar, operator >)
ISAAC_DECLARE_BINARY_OPERATOR(scalar, operator >=)
ISAAC_DECLARE_BINARY_OPERATOR(scalar, operator <)
ISAAC_DECLARE_BINARY_OPERATOR(scalar, operator <=)
ISAAC_DECLARE_BINARY_OPERATOR(scalar, operator ==)
ISAAC_DECLARE_BINARY_OPERATOR(scalar, operator !=)

ISAAC_DECLARE_BINARY_OPERATOR(scalar, (max))
ISAAC_DECLARE_BINARY_OPERATOR(scalar, (min))
ISAAC_DECLARE_BINARY_OPERATOR(scalar, pow)

#undef ISAAC_DECLARE_BINARY_OPERATOR

//--------------------------------

//Unary operators
#define ISAAC_DECLARE_UNARY_OPERATOR(OPNAME) \
scalar OPNAME (scalar const & x);\

ISAAC_DECLARE_UNARY_OPERATOR(abs)
ISAAC_DECLARE_UNARY_OPERATOR(acos)
ISAAC_DECLARE_UNARY_OPERATOR(asin)
ISAAC_DECLARE_UNARY_OPERATOR(atan)
ISAAC_DECLARE_UNARY_OPERATOR(ceil)
ISAAC_DECLARE_UNARY_OPERATOR(cos)
ISAAC_DECLARE_UNARY_OPERATOR(cosh)
ISAAC_DECLARE_UNARY_OPERATOR(exp)
ISAAC_DECLARE_UNARY_OPERATOR(floor)
ISAAC_DECLARE_UNARY_OPERATOR(log)
ISAAC_DECLARE_UNARY_OPERATOR(log10)
ISAAC_DECLARE_UNARY_OPERATOR(sin)
ISAAC_DECLARE_UNARY_OPERATOR(sinh)
ISAAC_DECLARE_UNARY_OPERATOR(sqrt)
ISAAC_DECLARE_UNARY_OPERATOR(tan)
ISAAC_DECLARE_UNARY_OPERATOR(tanh)
ISAAC_DECLARE_UNARY_OPERATOR(trans)

#undef ISAAC_DECLARE_UNARY_OPERATOR

ISAACAPI std::ostream & operator<<(std::ostream & os, scalar const & s);
ISAACAPI scalar cast(scalar const & in, numeric_type dtype);

}

#endif
