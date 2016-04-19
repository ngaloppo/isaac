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

#include "isaac/driver/program_cache.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/runtime/instruction.h"
#include "isaac/runtime/exceptions.h"
#include "isaac/runtime/inference/random_forest.h"

namespace isaac
{
namespace runtime
{


driver::Program const & instruction::init(expression_tree const & tree)
{
  driver::Context & context = (driver::Context&)tree.context();
  std::string pname = symbolic::hash(tree);
  driver::Program const * program = cache_.find(pname);
  if(program)
      return *program;
  std::string srcs;
   for(unsigned int i = 0 ; i < templates_.size() ; ++i)
     srcs += templates_[i]->generate(tools::to_string(i), tree, context.device());
   return cache_.add(context, pname, srcs);
}

instruction::instruction(inference::random_forest const & predictor, std::vector< std::shared_ptr<templates::base> > const & templates, driver::CommandQueue const & queue) :
  templates_(templates), predictor_(new inference::random_forest(predictor)), queue_(queue), cache_(driver::backend::programs::get(queue))
{
  cache_.clear();
}


instruction::instruction(std::shared_ptr<templates::base> const & tp, driver::CommandQueue const & queue) : templates_{tp}, queue_(queue), cache_(driver::backend::programs::get(queue))
{
  cache_.clear();
}

void instruction::execute(expression_tree const & tree, environment const & env, optimize const & opt)
{
  driver::Program const & program = init(tree);
  std::vector<int_t> x = templates_[0]->input_sizes(tree);
  static const int MAX_TEMPORARY_WORKSPACE = 1e6;

  //Specific tuning if requested
  if(opt.tune && hardcoded_.find(x)==hardcoded_.end())
  {
    std::vector<double> timings(templates_.size());
    for(unsigned int i = 0 ; i < templates_.size() ; ++i)
    {
      if(templates_[i]->temporary_workspace(tree) > MAX_TEMPORARY_WORKSPACE){
          timings[i] = INFINITY;
          continue;
      }
      std::list<driver::Event> events;
      try{
        templates_[i]->enqueue(queue_, program, tools::to_string(i), tree, runtime::environment(0, &events));
        queue_.synchronize();
        auto time_event = [&](long sum, driver::Event const & e){ return sum + e.elapsed_time(); };
        timings[i] = 1e-9*std::accumulate(events.begin(), events.end(), 0, time_event);
      }catch(...){
        timings[i] = INFINITY;
      }
    }
    //Fill the override
    std::vector<int_t> x = templates_[0]->input_sizes(tree);
    hardcoded_[x] = std::distance(timings.begin(),std::min_element(timings.begin(), timings.end()));
  }

  //Prediction
  int label = 0;
  if(opt.label>=0)
    label = opt.label;
  else  if(hardcoded_.find(x)!=hardcoded_.end())
    label = hardcoded_.at(x);
  else if(predictor_.get())
  {
    std::vector<float> predictions = predictor_->predict(x);
    do{
        label = std::distance(predictions.begin(),std::max_element(predictions.begin(), predictions.end()));
        predictions[label] = 0;
    }while(templates_[label]->temporary_workspace(tree) > MAX_TEMPORARY_WORKSPACE);
  }

  //Execution
  if(templates_[label]->temporary_workspace(tree) > MAX_TEMPORARY_WORKSPACE)
    throw runtime_error("Running this operation would require an overly large temporary.");

  return templates_[label]->enqueue(queue_, program, tools::to_string(label), tree, env);
}

}
}
