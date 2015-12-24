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

#include <cassert>
#include <iostream>
#include <set>
#include <string>

#include "isaac/tuple.h"
#include "isaac/kernels/symbolic_object.h"
#include "isaac/kernels/parse.h"
#include "isaac/kernels/stream.h"
#include "isaac/symbolic/expression.h"

#include "isaac/tools/cpp/string.hpp"

namespace isaac
{

namespace symbolic
{

lambda::lambda(std::string const & code): code_(code)
{
  size_t pos_po = code_.find('(');
  size_t pos_pe = code_.find(')');
  name_ = code.substr(0, pos_po);
  args_ = tools::split(code.substr(pos_po + 1, pos_pe - pos_po - 1), ',');
  tokens_ = tools::tokenize(code_.substr(code_.find(":") + 1), "()[],*+/- ");
}

lambda::lambda(const char *code) : lambda(std::string(code))
{

}

int lambda::expand(std::string & str) const
{
  size_t pos = 0;
  size_t num_touched = 0;
  while ((pos=str.find(name_, pos==0?0:pos + 1))!=std::string::npos)
  {
    size_t pos_po = str.find('(', pos);
    size_t pos_pe = str.find(')', pos_po);

    //Parse
    std::vector<std::string> args = tools::split(str.substr(pos_po + 1, pos_pe - pos_po - 1), ',');
    if(args_.size() != args.size()){
      pos = pos_pe;
      continue;
    }

    //Process
    std::vector<std::string> tokens = tokens_;
    for(size_t i = 0 ; i < args_.size() ; ++i)
      std::replace(tokens.begin(), tokens.end(), args_[i], args[i]);

    //Replace
    str.replace(pos, pos_pe + 1 - pos, tools::join(tokens.begin(), tokens.end(), ""));
    num_touched++;
  }
  return num_touched;
}

bool lambda::operator<(lambda const & o) const
{
  return std::make_tuple(name_, args_.size()) < std::make_tuple(o.name_, o.args_.size());
}

//
object::object(std::string const & scalartype, std::string const & name, std::string const & type) : type_(type)
{
  attributes_["scalartype"] = scalartype;
  attributes_["name"] = name;
}

object::object(std::string const & scalartype, unsigned int id, std::string const & type) : object(scalartype, "obj" + tools::to_string(id), type)
{

}

object::~object()
{

}

std::string object::type() const
{
  return type_;
}


std::string object::process(std::string const & in) const
{
  std::string res = in;
  //Macros
  bool modified;
  do{
    modified = false;
    for (auto const & key : lambdas_)
      modified = modified || key.expand(res);
  }while(modified);
  //Attributes
  for (auto const & key : attributes_)
    tools::find_and_replace(res, "#" + key.first, key.second);
  return res;
}

bool object::hasattr(std::string const & name) const
{
  return attributes_.find(name) != attributes_.end();
}

std::string object::evaluate(std::map<std::string, std::string> const & accessors) const
{
  if (accessors.find(type_)==accessors.end())
    return process("#name");
  return process(accessors.at(type_));
}

//
reduction::reduction(std::string const & scalartype, unsigned int id, size_t root, op_element op, std::string const & type) :
  object(scalartype, id, type), index_(root), op_(op)
{ }

size_t reduction::index() const
{ return index_; }

op_element reduction::op() const
{ return op_; }

//
reduce_1d::reduce_1d(std::string const & scalartype, unsigned int id, size_t root, op_element op) : reduction(scalartype, id, root, op, "reduce_1d"){ }

//
reduce_2d::reduce_2d(std::string const & scalartype, unsigned int id, size_t root, op_element op) : reduction(scalartype, id, root, op, "reduce_2d") { }

//
placeholder::placeholder(unsigned int level) : object("int", "sforidx" + tools::to_string(level), "placeholder"){}

//
host_scalar::host_scalar(std::string const & scalartype, unsigned int id) : object(scalartype, id, "host_scalar"){ }

//
array::array(std::string const & scalartype, unsigned int id, std::string const & type) : object(scalartype, id, type)
{
  attributes_["pointer"] = process("#name_pointer");
}

//
unsigned int buffer::_dim(tuple const & shape) const
{
  unsigned int result = 0;
  for(size_t i = 0 ; i < shape.size() ; ++i) result += shape[i]>1;
  return result;
}

buffer::buffer(std::string const & scalartype, unsigned int id, std::string const & type, const tuple &shape) : array(scalartype, id, type), dim_(_dim(shape))
{
  attributes_["start"] = process("#name_start");
  if(dim_ > 0) attributes_["stride"] = process("#name_stride");
  if(dim_ > 1) attributes_["ld"] = process("#name_ld");

  //Default
  if(dim_==0) lambdas_.insert("at(): #pointer[#start]");
  if(dim_==1) lambdas_.insert("at(i): #pointer[#start + (i)*#stride]");
  if(dim_==2) lambdas_.insert("at(i, j): #pointer[#start + (i)*#stride + (j)*#ld]");

  //Broadcast
  if(dim_!=shape.size())
  {
    std::string broadcast = "at(";
    for(size_t i = 0 ; i < shape.size() ; ++i)
      broadcast += ((broadcast.back()=='(')?"arg":",arg") + tools::to_string(i);
    broadcast += ") : at(";
    for(size_t i = 0 ; i < shape.size() ; ++i)
      if(shape[i] > 1)
        broadcast += ((broadcast.back()=='(')?"arg":",arg") + tools::to_string(i);
    broadcast += ")";
    lambdas_.insert(broadcast);
  }
}

//
index_modifier::index_modifier(const std::string &scalartype, unsigned int id, size_t index, mapping_type const & mapping, std::string const & type) : array(scalartype, id, type), index_(index), mapping_(mapping)
{ }

//
reshape::reshape(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping) : index_modifier(scalartype, id, index, mapping, "reshape")
{

}

//
diag_matrix::diag_matrix(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping) : index_modifier(scalartype, id, index, mapping, "vdiag"){}

//
array_access::array_access(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping) : index_modifier(scalartype, id, index, mapping, "array_access")
{ }

//
matrix_row::matrix_row(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping) : index_modifier(scalartype, id, index, mapping, "matrix_row")
{ }

//
matrix_column::matrix_column(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping) : index_modifier(scalartype, id, index, mapping, "matrix_column")
{ }

//
diag_vector::diag_vector(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping) : index_modifier(scalartype, id, index, mapping, "matrix_diag")
{ }

repeat::repeat(std::string const & scalartype, unsigned int id,  size_t index, mapping_type const & mapping) : index_modifier(scalartype, id, index, mapping, "repeat")
{ }

//
std::string cast::operator_to_str(operation_type type)
{
  switch(type)
  {
    case CAST_BOOL_TYPE : return "bool";
    case CAST_CHAR_TYPE : return "char";
    case CAST_UCHAR_TYPE : return "uchar";
    case CAST_SHORT_TYPE : return "short";
    case CAST_USHORT_TYPE : return "ushort";
    case CAST_INT_TYPE : return "int";
    case CAST_UINT_TYPE : return "uint";
    case CAST_LONG_TYPE : return "long";
    case CAST_ULONG_TYPE : return "ulong";
    case CAST_HALF_TYPE : return "half";
    case CAST_FLOAT_TYPE : return "float";
    case CAST_DOUBLE_TYPE : return "double";
    default : return "invalid";
  }
}

cast::cast(operation_type type, unsigned int id) : object(operator_to_str(type), id, "cast")
{ }

object& get(expression_tree::container_type const & tree, size_t root, mapping_type const & mapping, size_t idx)
{
  for(unsigned int i = 0 ; i < idx ; ++i){
      expression_tree::node node = tree[root];
      if(node.rhs.subtype==COMPOSITE_OPERATOR_TYPE)
        root = node.rhs.node_index;
      else
        return *(mapping.at(std::make_pair(root, RHS_NODE_TYPE)));
  }
  return *(mapping.at(std::make_pair(root, LHS_NODE_TYPE)));
}

}
}
