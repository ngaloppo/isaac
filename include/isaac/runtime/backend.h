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

#ifndef ISAAC_MODEL_DATABASE_H
#define ISAAC_MODEL_DATABASE_H

#include <map>
#include <memory>
#include "isaac/common.h"
#include "isaac/common/expression_type.h"
#include "isaac/driver/command_queue.h"
#include "isaac/driver/device.h"
#include "isaac/jit/generation/base.h"

namespace isaac
{

namespace templates
{
class base;
}

namespace runtime
{

class instruction;
typedef std::map<std::pair<expression_type, numeric_type>, std::shared_ptr<instruction> > implementation;

class ISAACAPI backend
{
public:
  class ISAACAPI implementations
  {
      typedef std::tuple<driver::Device::Type, driver::Device::Vendor, driver::Device::Architecture> database_key;
      typedef std::map<database_key, const char *> database_type;

  private:
      static templates::base* create(std::string const & template_name, std::vector<int> const & x);
      static void import(std::string const & fname, driver::CommandQueue const & queue);
      static implementation & init(driver::CommandQueue const & queue);

  public:
      static implementation & get(driver::CommandQueue const & queue);
      static void release();

  private:
      static const database_type database_;
      static std::map<driver::CommandQueue, implementation> cache_;
  };
};



}
}

#endif
