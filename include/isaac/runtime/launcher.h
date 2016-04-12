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

#ifndef _ISAAC_SYMBOLIC_HANDLER_H
#define _ISAAC_SYMBOLIC_HANDLER_H

#include "isaac/jit/syntax/expression/expression.h"

namespace isaac
{

namespace runtime
{

struct environment
{
  environment(unsigned int _queue_id = 0, std::list<driver::Event>* _events = NULL, std::vector<driver::Event>* _dependencies = NULL) :
     events(_events), dependencies(_dependencies), queue_id_(_queue_id)
  {}

  environment(driver::CommandQueue const & queue, std::list<driver::Event> *_events = NULL, std::vector<driver::Event> *_dependencies = NULL) :
      events(_events), dependencies(_dependencies), queue_id_(-1), queue_(new driver::CommandQueue(queue))
  {}

  void enqueue(driver::Context const & context, driver::Kernel const & kernel, driver::NDRange global, driver::NDRange local) const
  {
    driver::CommandQueue & q = queue(context);
    if(events)
    {
      driver::Event event(q.backend());
      q.enqueue(kernel, global, local, dependencies, &event);
      events->push_back(event);
    }
    else
      q.enqueue(kernel, global, local, dependencies, NULL);
  }

  driver::CommandQueue & queue(driver::Context const & context) const
  {
    if(queue_)
        return *queue_;
    return driver::backend::queues::get(context, queue_id_);
  }

  std::list<driver::Event>* events;
  std::vector<driver::Event>* dependencies;

private:
  int queue_id_;
  std::shared_ptr<driver::CommandQueue> queue_;
};

struct optimize
{
  optimize(bool _tune = false, int _label = -1) : tune(_tune), label(_label){}
  bool tune;
  int label;
};

class launcher
{
public:
  launcher(expression_tree const & tree, environment const& env = environment(), optimize const & opt = optimize()) : tree_(tree), env_(env), opt_(opt){}
  launcher(expression_tree const & tree, launcher const & other) : tree_(tree), env_(other.env_), opt_(other.opt_){}
  expression_tree const & tree() const { return tree_; }
  environment const & env() const { return env_; }
  optimize const & opt() const { return opt_; }
private:
  expression_tree tree_;
  environment env_;
  optimize opt_;
};

}
}

#endif
