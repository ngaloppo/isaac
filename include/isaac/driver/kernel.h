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

#ifndef ISAAC_DRIVER_KERNEL_H
#define ISAAC_DRIVER_KERNEL_H

#include "isaac/common.h"
#include "isaac/driver/dispatch.h"
#include "isaac/driver/handle.h"

#include <memory>

namespace isaac
{

class scalar;

namespace driver
{

class Buffer;
class Program;

// Kernel
class ISAACAPI Kernel
{
  friend class CommandQueue;
public:
  typedef HANDLE_TYPE(cl_kernel, CUfunction) handle_type;

public:
  Kernel(Program const & program, const char * name);
  void setArg(unsigned int index, scalar const & scal);
  void setArg(unsigned int index, size_t size, void* ptr);
  void setArg(unsigned int index, Buffer const &);
  void setSizeArg(unsigned int index, size_t N);
  template<class T> void setArg(unsigned int index, T value) { setArg(index, sizeof(T), (void*)&value); }

private:
  backend_type backend_;
  unsigned int address_bits_;
  std::vector<std::shared_ptr<void> >  cu_params_store_;
  std::vector<void*>  cu_params_;
  handle_type h_;
};

}

}

#endif

