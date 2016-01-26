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

#ifndef ISAAC_MAPPED_OBJECT_H
#define ISAAC_MAPPED_OBJECT_H

#include <set>
#include <map>
#include <string>
#include "isaac/types.h"
#include "isaac/templates/engine/stream.h"
#include "isaac/symbolic/expression/expression.h"

namespace isaac
{

namespace symbolic
{

class object;

typedef std::map<size_t, std::shared_ptr<object> > symbols_table;

//Lambda
class lambda
{
public:
  lambda(std::string const & code);
  lambda(const char * code);
  int expand(std::string & str) const;
  bool operator<(lambda const & o) const;

private:
  std::string code_;
  std::string name_;
  std::vector<std::string> args_;
  std::vector<std::string> tokens_;
};

//Objects
class object
{
public:
  object(std::string const & scalartype, std::string const & name);
  object(std::string const & scalartype, unsigned int id);
  virtual ~object();
  std::string process(std::string const & in) const;
  virtual std::string evaluate(std::string const &) const;
  bool hasattr(std::string const & name) const;
protected:
  std::map<std::string, std::string> attributes_;
  std::set<lambda> lambdas_;
};

//Binary Node
class binary_node
{
public:
  binary_node(size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
  op_element op() const;
  object const * lhs() const;
  object const * rhs() const;
  size_t root() const;
protected:
  op_element op_;
  std::string op_str_;
  object* lhs_;
  object* rhs_;
  size_t root_;
};

//Sfor
class sfor: public object, public binary_node
{
public:
  sfor(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
};

//Binary arithmetic
class binary_arithmetic_node: public object, public binary_node
{
public:
  binary_arithmetic_node(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
  std::string evaluate(std::string const &) const;
};

//Unary arithmetic
class unary_arithmetic_node: public object, public binary_node
{
public:
  unary_arithmetic_node(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
  std::string evaluate(std::string const &) const;
};

//Reductions
class reduction : public object, public binary_node
{
public:
  reduction(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
};

class reduce_1d : public reduction
{
public:
  reduce_1d(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
};

class reduce_2d : public reduction
{
public:
  reduce_2d(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
};

//Host scalar
class host_scalar : public object
{
public:
  host_scalar(std::string const & scalartype, unsigned int id);
  std::string evaluate(std::string const &) const;
};

//Placeholder
class placeholder : public object
{
public:
  placeholder(unsigned int level);
  std::string evaluate(std::string const &) const;
};

//Arrays
class array : public object
{
protected:
  std::string make_broadcast(tuple const & shape);
public:
  array(std::string const & scalartype, unsigned int id);
};

//Buffer
class buffer : public array
{
public:
  buffer(std::string const & scalartype, unsigned int id, tuple const & shape);
  unsigned int dim() const { return dim_; }
private:
  std::string ld_;
  std::string start_;
  std::string stride_;
  unsigned int dim_;
};

//Index modifier
class index_modifier: public array, public binary_node
{
public:
  index_modifier(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
};

class reshape : public index_modifier
{
public:
  reshape(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
};

class diag_matrix : public index_modifier
{
public:
  diag_matrix(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
};

class diag_vector : public index_modifier
{
public:
  diag_vector(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
};

class array_access: public index_modifier
{
public:
  array_access(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
};

class matrix_row : public index_modifier
{
public:
  matrix_row(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
};

class matrix_column : public index_modifier
{
public:
  matrix_column(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
};

class repeat : public index_modifier
{
//private:
//  static char get_type(node_info const & info);
public:
  repeat(std::string const & scalartype, unsigned int id, size_t root, op_element op, expression_tree const & tree, symbols_table const & table);
private:
  char type_;
};



}

}
#endif
