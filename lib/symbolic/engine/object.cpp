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

#include <string>

#include "isaac/array.h"
#include "isaac/exception/operation_not_supported.h"
#include "isaac/symbolic/engine/object.h"
#include "isaac/symbolic/expression/expression.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{
namespace symbolic
{

//
void object::add_base(const std::string &name)
{ hierarchy_.push_front(name); }

object::object(std::string const & scalartype, unsigned int id): object(scalartype, "obj" + tools::to_string(id))
{}

object::object(std::string const & scalartype, std::string const & name)
{
  add_base("object");

  //attributes
  attributes_["scalartype"] = scalartype;
  attributes_["name"]       = name;

  //macros
  macros_.insert("vload(i): at(i)");
  macros_.insert("vload2(i): (#scalartype2)(at(i), at(i+1))");
  macros_.insert("vload4(i): (#scalartype4)(at(i), at(i+1), at(i+2), at(i+3))");

  macros_.insert("vload(i,j): at(i,j)");
  macros_.insert("vload2(i,j): (#scalartype2)(at(i,j), at(i+1,j))");
  macros_.insert("vload4(i,j): (#scalartype4)(at(i,j), at(i+1,j), at(i+2,j), at(i+3,j))");
}

object::~object()
{}

std::string object::process(std::string const & in) const
{
  std::string res = in;
  //Macros
  bool modified;
  do{
    modified = false;
    for (auto const & key : macros_)
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

std::string object::evaluate(std::map<std::string, std::string> const & table) const
{
  for(std::string const & type: hierarchy_)
      for (auto const& supplied : table )
          if(type==supplied.first)
              return process(supplied.second);
  throw "NOT FOUND";
}

//
leaf::leaf(std::string const & scalartype, unsigned int id): object(scalartype, id)
{ add_base("leaf"); }

leaf::leaf(std::string const & scalartype, std::string const & name): object(scalartype, name)
{ add_base("leaf"); }

//
node::node(size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : op_(op), lhs_(NULL), rhs_(NULL), root_(root)
{
  expression_tree::node const & node = tree[root];
  symbols_table::const_iterator it;
  if((it = table.find(node.binary_operator.lhs))!=table.end())
    lhs_ = it->second.get();
  if((it = table.find(node.binary_operator.rhs))!=table.end())
    rhs_ = it->second.get();
}

op_element node::op() const
{ return op_; }

object const * node::lhs() const
{ return lhs_; }

object const * node::rhs() const
{ return rhs_; }

size_t node::root() const
{ return root_; }

//
sfor::sfor(unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : object(to_string(tree[root].dtype), id), node(root, op, tree, table)
{ add_base("sfor"); }

//
arithmetic_node::arithmetic_node(unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : object(to_string(tree[root].dtype), id), node(root, op, tree, table), op_str_(to_string(op.type))
{ }

//
binary_arithmetic_node::binary_arithmetic_node(unsigned int id, size_t root, op_element op, expression_tree const & expression, symbols_table const & table) : arithmetic_node(id, root, op, expression, table)
{ add_base("binary_arithmetic_node"); }

std::string binary_arithmetic_node::evaluate(std::map<std::string, std::string> const & table) const
{
  std::string arg0 = lhs_->evaluate(table);
  std::string arg1 = rhs_->evaluate(table);
  if(is_function(op_.type))
    return op_str_ + "(" + arg0 + ", " + arg1 + ")";
  else
    return "(" + arg0 + op_str_ + arg1 + ")";
}

//
unary_arithmetic_node::unary_arithmetic_node(unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) :
    arithmetic_node(id, root, op, tree, table)
{ add_base("unary_arithmetic_node"); }

std::string unary_arithmetic_node::evaluate(std::map<std::string, std::string> const & table) const
{ return op_str_ + "(" + lhs_->evaluate(table) + ")"; }

//
reduction::reduction(unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) :
  object(to_string(tree[root].dtype), id), node(root, op, tree, table)
{ add_base("reduction"); }

//
reduce_1d::reduce_1d(unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : reduction(id, root, op, tree, table)
{ add_base("reduce_1d"); }


//
reduce_2d::reduce_2d(unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : reduction(id, root, op, tree, table)
{ add_base("reduce_2d"); }

//
placeholder::placeholder(unsigned int level) : leaf("int", "sforidx" + tools::to_string(level))
{
  macros_.insert("at(): #name");
  macros_.insert("at(i): #name");
  macros_.insert("at(i,j): #name");

  add_base("placebolder");
}

//
host_scalar::host_scalar(std::string const & scalartype, unsigned int id) : leaf(scalartype, id)
{
  macros_.insert("at(): #name_value");
  macros_.insert("at(i): #name_value");
  macros_.insert("at(i,j): #name_value");

  add_base("host_scalar");
}

//
array::array(std::string const & scalartype, unsigned int id) : leaf(scalartype, id)
{
  attributes_["pointer"] = process("#name_pointer");

  add_base("array");
}

std::string array::make_broadcast(const tuple &shape)
{
  std::string result = "at(";
  for(size_t i = 0 ; i < shape.size() ; ++i)
    result += ((result.back()=='(')?"arg":",arg") + tools::to_string(i);
  result += ") : at(";
  for(size_t i = 0 ; i < shape.size() ; ++i)
    if(shape[i] > 1)
      result += ((result.back()=='(')?"arg":",arg") + tools::to_string(i);
  result += ")";
  return result;
}

//
buffer::buffer(std::string const & scalartype, unsigned int id, const tuple &shape) : array(scalartype, id), dim_(numgt1(shape))
{
  //Attributes
  attributes_["off"] = process("#name_off");
  for(unsigned int i = 0 ; i < dim_ ; ++i){
    std::string inc = "inc" + tools::to_string(i);
    attributes_[inc] = process("#name_" + inc);
  }

  //Access
  std::vector<std::string> args;
  for(unsigned int i = 0 ; i < dim_ ; ++i)
    args.push_back("x" + tools::to_string(i));

  std::string off = "#off";
  for(unsigned int i = 0 ; i < dim_ ; ++i)
  {
    std::string inc = "#inc" + tools::to_string(i);
    off += " + (" + args[i] + ")*" + inc;
  }
  macros_.insert("at(" + tools::join(args, ",") + "): #pointer[" + off + "]");

  //Broadcast
  if(dim_!=shape.size())
    macros_.insert(make_broadcast(shape));

  add_base("buffer");
}

//
index_modifier::index_modifier(const std::string &scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : array(scalartype, id), node(root, op, tree, table)
{ add_base("index_modifier"); }

//Reshaping
reshape::reshape(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : index_modifier(scalartype, id, root, op, tree, table)
{
  add_base("reshape");

  tuple new_shape = tree[root].shape;
  tuple old_shape = tree[tree[root].binary_operator.lhs].shape;

  //Attributes
  for(unsigned int i = 1 ; i < new_shape.size() ; ++i)
    if(new_shape[i] > 1){
      std::string inc = "new_inc" + tools::to_string(i);
      attributes_[inc] = process("#name_" + inc);
    }

  for(unsigned int i = 1 ; i < old_shape.size() ; ++i)
    if(old_shape[i] > 1){
      std::string inc = "old_inc" + tools::to_string(i);
      attributes_[inc] = process("#name_" + inc);
    }

  //Index modification
  size_t new_gt1 = numgt1(new_shape);
  size_t old_gt1 = numgt1(old_shape);

  if(new_gt1==0 && old_gt1==0)
    macros_.insert("at(): " + lhs_->evaluate({{"leaf","at()"}}));
  if(new_gt1==0 && old_gt1==1)
    macros_.insert("at(): " + lhs_->evaluate({{"leaf","at(0)"}}));
  if(new_gt1==0 && old_gt1==2)
    macros_.insert("at(): " + lhs_->evaluate({{"leaf","at(0,0)"}}));

  if(new_gt1==1 && old_gt1==0)
    macros_.insert("at(i): " + lhs_->evaluate({{"leaf","at()"}}));
  if(new_gt1==1 && old_gt1==1)
    macros_.insert("at(i): " + lhs_->evaluate({{"leaf","at(i)"}}));
  if(new_gt1==1 && old_gt1==2)
    macros_.insert("at(i): " + lhs_->evaluate({{"leaf","at((i)%#old_inc1, (i)/#old_inc1)"}}));

  if(new_gt1==2 && old_gt1==0)
    macros_.insert("at(i,j): " + lhs_-> evaluate({{"leaf","at()"}}));
  if(new_gt1==2 && old_gt1==1)
    macros_.insert("at(i,j): " + lhs_-> evaluate({{"leaf","at((i) + (j)*#new_inc1)"}}));
  if(new_gt1==2 && old_gt1==2)
    macros_.insert("at(i,j): " + lhs_->evaluate({{"leaf","at(((i) + (j)*#new_inc1)%#old_inc1, ((i)+(j)*#new_inc1)/#old_inc1)"}}));

  //Broadcast
  if(new_gt1!=new_shape.size())
    macros_.insert(make_broadcast(new_shape));
}

//
diag_vector::diag_vector(const std::string &scalartype, unsigned int id, size_t root, op_element op, const expression_tree &tree, const symbols_table &table) : index_modifier(scalartype, id, root, op, tree, table)
{
  add_base("diag_vector");

  macros_.insert("at(i,j): " + lhs_->evaluate({{"leaf","(i==j)?at(i):0"}}));
  tuple const & shape = tree[root].shape;
  if(numgt1(shape)!=shape.size())
    macros_.insert(make_broadcast(shape));
}

////

}
}
