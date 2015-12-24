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
#include "isaac/kernels/stream.h"
#include "isaac/symbolic/expression.h"

namespace isaac
{

enum leaf_t
{
  LHS_NODE_TYPE,
  PARENT_NODE_TYPE,
  RHS_NODE_TYPE
};

namespace symbolic
{

class object;

typedef std::pair<size_t, leaf_t> mapping_key;
typedef std::map<mapping_key, std::shared_ptr<object> > mapping_type;

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
  object(std::string const & scalartype, std::string const & name, std::string const & type);
  object(std::string const & scalartype, unsigned int id, std::string const & type);
  virtual ~object();

  std::string type() const;
  std::string process(std::string const & in) const;
  std::string evaluate(std::map<std::string, std::string> const & accessors) const;
  bool hasattr(std::string const & name) const;
protected:
  std::string type_;
  std::map<std::string, std::string> attributes_;
  std::set<lambda> lambdas_;
};

class reduction : public object
{
public:
  reduction(std::string const & scalartype, unsigned int id, size_t root, op_element op, std::string const & type);
  size_t index() const;
  op_element op() const;
private:
  size_t index_;
  op_element op_;
};

class reduce_1d : public reduction
{
public:
  reduce_1d(std::string const & scalartype, unsigned int id, size_t root, op_element op);
};

class reduce_2d : public reduction
{
public:
  reduce_2d(std::string const & scalartype, unsigned int id, size_t root, op_element op);
};

class host_scalar : public object
{
public:
  host_scalar(std::string const & scalartype, unsigned int id);
};

class placeholder : public object
{
public:
  placeholder(unsigned int level);
};

class array : public object
{
public:
  array(std::string const & scalartype, unsigned int id, std::string const & type);
};

class buffer : public array
{
private:
  unsigned int _dim(tuple const & shape) const;
public:
  buffer(std::string const & scalartype, unsigned int id, std::string const & type, tuple const & shape);
  unsigned int dim() const { return dim_; }
private:
  std::string ld_;
  std::string start_;
  std::string stride_;
  unsigned int dim_;
};

class index_modifier: public array
{
public:
  index_modifier(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping, std::string const & type);
private:
  size_t index_;
  mapping_type const & mapping_;
};

class reshape : public index_modifier
{
public:
  reshape(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping);
};

class diag_matrix : public index_modifier
{
public:
  diag_matrix(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping);
};

class diag_vector : public index_modifier
{
public:
  diag_vector(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping);
};

class array_access: public index_modifier
{
public:
  array_access(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping);
};

class matrix_row : public index_modifier
{
public:
  matrix_row(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping);
};

class matrix_column : public index_modifier
{
public:
  matrix_column(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping);
};

class repeat : public index_modifier
{
//private:
//  static char get_type(node_info const & info);
public:
  repeat(std::string const & scalartype, unsigned int id, size_t index, mapping_type const & mapping);
private:
  char type_;
};


class cast : public object
{
  static std::string operator_to_str(operation_type type);
public:
  cast(operation_type type, unsigned int id);
};

extern object& get(expression_tree::container_type const &, size_t, symbolic::mapping_type const &, size_t);

}

}
#endif
