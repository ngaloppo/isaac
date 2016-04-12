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

#include <fstream>
#include <algorithm>
#include <memory>
#include <numeric>

#include "rapidjson/document.h"
#include "rapidjson/to_array.hpp"

#include "isaac/driver/program_cache.h"
#include "isaac/runtime/backend.h"
#include "isaac/jit/generation/elementwise_1d.h"
#include "isaac/jit/generation/reduce_1d.h"
#include "isaac/jit/generation/elementwise_2d.h"
#include "isaac/jit/generation/reduce_2d.h"
#include "isaac/jit/generation/matrix_product.h"
#include "isaac/exception/api.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/tools/sys/getenv.hpp"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{
namespace runtime
{

templates::base* backend::implementations::create(std::string const & name, std::vector<int> const & p)
{
  using namespace templates;
  fetching_policy_type fetch[] = {FETCH_FROM_LOCAL, FETCH_FROM_GLOBAL_STRIDED, FETCH_FROM_GLOBAL_CONTIGUOUS};
  if(name=="elementwise_1d")
    return new elementwise_1d(p[0], p[1], p[2], fetch[p[3]]);
  else if(name=="reduce_1d")
    return new reduce_1d(p[0], p[1], p[2], fetch[p[3]]);
  else if(name=="elementwise_2d")
    return new elementwise_2d(p[0], p[1], p[2], p[3], p[4], fetch[p[5]]);
  else if(name.find("reduce_2d_rows")!=std::string::npos)
    return new reduce_2d_rows(p[0], p[1], p[2], p[3], p[4], fetch[p[5]]);
  else if(name.find("reduce_2d_cols")!=std::string::npos)
    return new reduce_2d_cols(p[0], p[1], p[2], p[3], p[4], fetch[p[5]]);
  else if(name.find("matrix_product_nn")!=std::string::npos)
    return new matrix_product_nn(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], fetch[p[8]], fetch[p[9]], p[10], p[11]);
  else if(name.find("matrix_product_tn")!=std::string::npos)
    return new matrix_product_tn(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], fetch[p[8]], fetch[p[9]], p[10], p[11]);
  else if(name.find("matrix_product_nt")!=std::string::npos)
    return new matrix_product_nt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], fetch[p[8]], fetch[p[9]], p[10], p[11]);
  else if(name.find("matrix_product_tt")!=std::string::npos)
    return new matrix_product_tt(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], fetch[p[8]], fetch[p[9]], p[10], p[11]);
  else
    throw std::invalid_argument("Invalid expression: " + name);
}

void backend::implementations::import(std::string const & str, driver::CommandQueue const & queue)
{
  implementation & result = cache_[queue];
  //Parse the JSON document
  rapidjson::Document document;
  document.Parse<0>(str.c_str());
  //Deserialize
  std::vector<std::string> operations = {"elementwise_1d", "reduce_1d", "elementwise_2d", "reduce_2d_rows", "reduce_2d_cols", "matrix_product_nn", "matrix_product_tn", "matrix_product_nt", "matrix_product_tt"};
  std::vector<std::string> dtype = {"float32", "float64"};
  for(auto & operation : operations)
  {
    const char * opcstr = operation.c_str();
    if(document.HasMember(opcstr))
    {
      expression_type etype = expression_type_from_string(operation);
      for(auto & elem : dtype)
      {
        const char * dtcstr = elem.c_str();
        if(document[opcstr].HasMember(dtcstr))
        {
          numeric_type dtype = numeric_type_from_string(elem);
          // Get profiles
          std::vector<std::shared_ptr<templates::base> > templates;
          rapidjson::Value const & profiles = document[opcstr][dtcstr]["profiles"];
          for (rapidjson::SizeType id = 0 ; id < profiles.Size() ; ++id)
            templates.emplace_back(create(operation, rapidjson::to_int_array<int>(profiles[id])));
          if(templates.size()>1)
          {
            // Get predictor
            inference::random_forest predictor(document[opcstr][dtcstr]["predictor"]);
            result[std::make_pair(etype, dtype)] = std::make_shared<instruction>(predictor, templates, queue);
          }
          else
            result[std::make_pair(etype, dtype)] = std::make_shared<instruction>(templates[0], queue);
        }
      }
    }
  }
}

implementation& backend::implementations::init(driver::CommandQueue const & queue)
{
  implementation & map = cache_[queue];
  driver::Device const & device = queue.device();
  database_type::const_iterator it = database_.find(std::make_tuple(device.type(), device.vendor(), device.architecture()));
  /*-- Device not found in database --*/
  if(it==database_.end()){
      import(database_.at(std::make_tuple(driver::Device::Type::UNKNOWN, driver::Device::Vendor::UNKNOWN, driver::Device::Architecture::UNKNOWN)), queue);
  }
  /*-- Device found in database --*/
  else{
      import(it->second, queue);
  }

  /*-- User-provided profile --*/
  std::string homepath = tools::getenv("HOME");
  if(homepath.size())
  {
    std::string json_path = homepath + "/.isaac/devices/device0.json";
    std::ifstream t(json_path);
    if(!t)
        return map;
    std::string str;
    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);
    str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    import(str, queue);
  }

  return map;
}

implementation& backend::implementations::get(driver::CommandQueue const & queue)
{
  std::map<driver::CommandQueue, implementation>::iterator it = cache_.find(queue);
  if(it == cache_.end())
    return init(queue);
  return it->second;
}

void backend::implementations::release()
{
  cache_.clear();
}

std::map<driver::CommandQueue, implementation> backend::implementations::cache_;

}
}
