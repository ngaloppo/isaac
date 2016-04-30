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

#include <assert.h>
#include <list>
#include <vector>
#include <stdexcept>
#include "isaac/array.h"
#include "isaac/jit/compile.h"
#include "isaac/jit/syntax/expression/preset.h"
#include "isaac/runtime/execute.h"
#include "isaac/runtime/instruction.h"
#include "isaac/expression.h"

namespace isaac
{
namespace runtime
{
  void execute(launcher const &, implementation &)
  { }

  void execute(launcher const & c)
  { execute(c, backend::implementations::get(c.env().queue(c.tree().context()))); }

}
}
