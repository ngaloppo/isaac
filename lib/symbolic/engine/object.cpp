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
    size_t next = str.find('(', pos_po + 1);
    while(next < pos_pe){
      pos_pe = str.find(')', pos_pe + 1);
      if(next < pos_pe)
        next = str.find('(', next + 1);
    }
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
object::object(std::string const & scalartype, std::string const & name)
{
  attributes_["scalartype"] = scalartype;
  attributes_["name"] = name;
}

object::object(std::string const & scalartype, unsigned int id) : object(scalartype, "obj" + tools::to_string(id))
{

}

object::~object()
{

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

std::string object::evaluate(const std::string & str) const
{ return process(str); }

//
binary_node::binary_node(size_t root, op_element op, expression_tree const & expression, symbols_table const & table) : op_(op), op_str_(to_string(op.type)), lhs_(NULL), rhs_(NULL)
{
  expression_tree::node const & node = expression[root];
  symbols_table::const_iterator it;
  if((it = table.find(node.binary_operator.lhs))!=table.end())
    lhs_ = it->second.get();
  if((it = table.find(node.binary_operator.rhs))!=table.end())
    rhs_ = it->second.get();
}

op_element binary_node::op() const
{ return op_; }

object const * binary_node::lhs() const
{ return lhs_; }

object const * binary_node::rhs() const
{ return rhs_; }

size_t binary_node::root() const
{ return root_; }

//sfor
sfor::sfor(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & expression, symbols_table const & table) :
  object(scalartype, id), binary_node(root, op, expression, table)
{

}

//
binary_arithmetic_node::binary_arithmetic_node(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & expression, symbols_table const & table) :
  object(scalartype, id), binary_node(root, op, expression, table){}

std::string binary_arithmetic_node::evaluate(std::string const & str) const
{
  std::string arg0 = lhs_->evaluate(str);
  std::string arg1 = rhs_->evaluate(str);
  if(is_function(op_.type))
    return op_str_ + "(" + arg0 + ", " + arg1 + ")";
  else
    return "(" + arg0 + op_str_ + arg1 + ")";
}

//
unary_arithmetic_node::unary_arithmetic_node(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) :
  object(scalartype, id), binary_node(root, op, tree, table){}

std::string unary_arithmetic_node::evaluate(std::string const & str) const
{ return op_str_ + "(" + lhs_->evaluate(str) + ")"; }

//
reduction::reduction(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) :
  object(scalartype, id), binary_node(root, op, tree, table)
{ }

//
reduce_1d::reduce_1d(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : reduction(scalartype, id, root, op, tree, table){ }

//
reduce_2d::reduce_2d(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : reduction(scalartype, id, root, op, tree, table) { }

//
placeholder::placeholder(unsigned int level) : object("int", "sforidx" + tools::to_string(level)){}

std::string placeholder::evaluate(std::string const &) const
{ return process("#name"); }

//
host_scalar::host_scalar(std::string const & scalartype, unsigned int id) : object(scalartype, id){ }

std::string host_scalar::evaluate(std::string const &) const
{ return process("#name"); }

//
array::array(std::string const & scalartype, unsigned int id) : object(scalartype, id)
{
  attributes_["pointer"] = process("#name_pointer");
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
  lambdas_.insert("at(" + tools::join(args, ",") + "): #pointer[" + off + "]");

  //Broadcast
  if(dim_!=shape.size())
    lambdas_.insert(make_broadcast(shape));
}

//
index_modifier::index_modifier(const std::string &scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : array(scalartype, id), binary_node(root, op, tree, table)
{ }

//Reshaping
reshape::reshape(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : index_modifier(scalartype, id, root, op, tree, table)
{
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
  if(new_gt1==1 && old_gt1==1)
    lambdas_.insert("at(i): " + lhs_->process("at(i)"));
  if(new_gt1==1 && old_gt1==2)
    lambdas_.insert("at(i): " + lhs_->process("at(i%#old_inc1, i/#old_inc1)"));
  if(new_gt1==2 && old_gt1==1)
    lambdas_.insert("at(i,j): " + lhs_-> process("at(i + j*#new_inc1)"));
  if(new_gt1==2 && old_gt1==2)
    lambdas_.insert("at(i,j): " + lhs_->process("at((i + j*#new_inc1)%#old_inc1, (i+j*#new_inc1)/#old_inc1)"));

  //Broadcast
  if(new_gt1!=new_shape.size())
    lambdas_.insert(make_broadcast(new_shape));
}

//
diag_matrix::diag_matrix(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : index_modifier(scalartype, id, root, op, tree, table){}

//
array_access::array_access(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : index_modifier(scalartype, id, root, op, tree, table)
{ }

//
matrix_row::matrix_row(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : index_modifier(scalartype, id, root, op, tree, table)
{ }

//
matrix_column::matrix_column(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : index_modifier(scalartype, id, root, op, tree, table)
{ }

//
diag_vector::diag_vector(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : index_modifier(scalartype, id, root, op, tree, table)
{ }

repeat::repeat(std::string const & scalartype, unsigned int id,  size_t root, op_element op, expression_tree const & tree, symbols_table const & table) : index_modifier(scalartype, id, root, op, tree, table)
{ }

////

}
}
