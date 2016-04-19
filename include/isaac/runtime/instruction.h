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

#ifndef ISAAC_RUNTIME_IMPLEMENTATION_H
#define ISAAC_RUNTIME_IMPLEMENTATION_H

#include <map>
#include <memory>
#include "isaac/common.h"
#include "isaac/driver/command_queue.h"
#include "isaac/jit/generation/base.h"

namespace isaac
{

class expression;

namespace runtime
{
namespace inference
{
  class random_forest;
}

class instruction
{
  typedef std::shared_ptr<templates::base> template_pointer;
  typedef std::vector<template_pointer> templates_container;

private:
  driver::Program const & init(expression const & tree);

public:
  instruction(inference::random_forest const &, std::vector< std::shared_ptr<templates::base> > const &, driver::CommandQueue const &);
  instruction(std::shared_ptr<templates::base> const &, driver::CommandQueue const &);
  void execute(expression const & tree, environment const & env, optimize const & opt);

private:
  templates_container templates_;
  std::shared_ptr<inference::random_forest> predictor_;
  std::map<std::vector<int_t>, int> hardcoded_;
  driver::CommandQueue queue_;
  driver::ProgramCache & cache_;
};


}
}

#endif
