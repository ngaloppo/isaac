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

#ifndef ISAAC_TEMPLATES_base_
#define ISAAC_TEMPLATES_base_


#include <list>
#include <set>
#include <cmath>

#include "isaac/jit/syntax/engine/object.h"
#include "isaac/runtime/launcher.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{

namespace templates
{

enum fetching_policy_type
{
  FETCH_FROM_LOCAL,
  FETCH_FROM_GLOBAL_STRIDED,
  FETCH_FROM_GLOBAL_CONTIGUOUS
};

//Error codes
static const int TEMPLATE_VALID = 0;
static const int TEMPLATE_LOCAL_MEMORY_OVERFLOW = -1;
static const int TEMPLATE_WORK_GROUP_SIZE_OVERFLOW = -2;
static const int TEMPLATE_LOCAL_SIZE_0_OVERFLOW = -3;
static const int TEMPLATE_LOCAL_SIZE_1_OVERFLOW = -4;
static const int TEMPLATE_LOCAL_SIZE_2_OVERFLOW = -5;
static const int TEMPLATE_LOCAL_SIZE_NOT_WARP_MULTIPLE = -6;
static const int TEMPLATE_INVALID_SIMD_WIDTH = -7;
static const int TEMPLATE_ALIGNMENT_MUST_BE_BLOCK_SIZE_MULTIPLE = -8;
static const int TEMPLATE_INVALID_FETCHING_POLICY_TYPE= -9;

static const int TEMPLATE_GLOBAL_MEMORY_REQUIRES_ZERO_LOCAL_FETCH = -10;
static const int TEMPLATE_MS_NS_MUST_BE_SIMD_WIDTH_MULTIPLE = -11;
static const int TEMPLATE_KS_MUST_BE_SMALLER_THAN_KL = -12;
static const int TEMPLATE_SIMD_WIDTH_MUST_BE_ONE = -13;
static const int TEMPLATE_LOCAL_FETCH_PRODUCT_MUST_MATCH_LOCAL_SIZE_PRODUCT = -14;
static const int TEMPLATE_LOCAL_FETCH_0_MUST_BE_KL_MULTIPLE = -15;
static const int TEMPLATE_LOCAL_FETCH_0_MUST_BE_NL_MULTIPLE = -16;
static const int TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE = -17;
static const int TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE = -18;
static const int TEMPLATE_TEMPORARY_TOO_LARGE = -19;
static const int TEMPLATE_BLOCK_SIZE_TOO_LARGE = -20;

class base
{
public:
  class genstream : public std::ostream
  {
  private:
    class buf : public std::stringbuf
    {
    public:
      buf(std::ostringstream& oss,size_t const & tab_count) ;
      int sync();
      ~buf();
    private:
      std::ostream& oss_;
      size_t const & tab_count_;
    };

  private:
    void process(std::string& str);

  public:
    genstream(driver::backend_type backend);
    ~genstream();
    std::string str();
    void inc_tab();
    void dec_tab();

  private:
    size_t tab_count_;
    driver::backend_type backend_;
    std::ostringstream oss;
  };

protected:

  template<class Fun>
  static inline void for_loop(genstream & stream, fetching_policy_type policy, unsigned int simd_width,
                              std::string const & idx, std::string const & bound, std::string const & domain_id, std::string const & domain_size, driver::Device const & device,
                              Fun const & generate_body)
  {
    std::string strwidth = tools::to_string(simd_width);

    std::string init, upper_bound, inc;
    //Loop infos
    if (policy==FETCH_FROM_GLOBAL_STRIDED)
    {
      init = domain_id;
      upper_bound = bound;
      inc = domain_size;
    }
    else if (policy==FETCH_FROM_GLOBAL_CONTIGUOUS)
    {
      std::string chunk_size = "chunk_size";
      std::string chunk_start = "chunk_start";
      std::string chunk_end = "chunk_end";

      stream << "$SIZE_T " << chunk_size << " = (" << bound << "+" << domain_size << "-1)/" << domain_size << ";" << std::endl;
      stream << "$SIZE_T " << chunk_start << " =" << domain_id << "*" << chunk_size << ";" << std::endl;
      stream << "$SIZE_T " << chunk_end << " = min(" << chunk_start << "+" << chunk_size << ", " << bound << ");" << std::endl;
      init = chunk_start;
      upper_bound = chunk_end;
      inc = "1";
    }

    //Actual loop
    std::string boundround = upper_bound + "/" + strwidth + "*" + strwidth;
    stream << "for(unsigned int " << idx << " = " << init << "*" << strwidth << "; " << idx << " < " << boundround << "; " << idx << " += " << inc << "*" << strwidth << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    generate_body(simd_width);
    stream.dec_tab();
    stream << "}" << std::endl;

    if (simd_width>1)
    {
      stream << "for(unsigned int " << idx << " = " << boundround << " + " << domain_id << "; " << idx << " < " << bound << "; " << idx << " += " + domain_size + ")" << std::endl;
      stream << "{" << std::endl;
      stream.inc_tab();
      generate_body(1);
      stream.dec_tab();
      stream << "}" << std::endl;
    }
  }

  static inline void compute_reduce_1d(genstream & os, std::string acc, std::string cur, token const & op)
  {
    if (is_function(op.type))
      os << acc << "=" << to_string(op.type) << "(" << acc << "," << cur << ");" << std::endl;
    else
      os << acc << "= (" << acc << ")" << to_string(op.type)  << "(" << cur << ");" << std::endl;
  }

  static inline void compute_index_reduce_1d(genstream & os, std::string acc, std::string cur, std::string const & acc_value, std::string const & cur_value, token const & op)
  {
    os << acc << " = " << cur_value << ">" << acc_value  << "?" << cur << ":" << acc << ";" << std::endl;
    os << acc_value << "=";
    if (op.type==ELEMENT_ARGFMAX_TYPE) os << "fmax";
    if (op.type==ELEMENT_ARGMAX_TYPE) os << "max";
    if (op.type==ELEMENT_ARGFMIN_TYPE) os << "fmin";
    if (op.type==ELEMENT_ARGMIN_TYPE) os << "min";
    os << "(" << acc_value << "," << cur_value << ");"<< std::endl;
  }

  static inline std::string neutral_element(token const & op, driver::backend_type backend, std::string const & dtype)
  {
    std::string INF = (backend==driver::OPENCL)?"INFINITY":"infinity<" + dtype + ">()";
    std::string N_INF = "-" + INF;
    switch (op.type)
    {
      case ADD_TYPE : return "0";
      case MULT_TYPE : return "1";
      case DIV_TYPE : return "1";
      case ELEMENT_FMAX_TYPE : return N_INF;
      case ELEMENT_ARGFMAX_TYPE : return N_INF;
      case ELEMENT_MAX_TYPE : return N_INF;
      case ELEMENT_ARGMAX_TYPE : return N_INF;
      case ELEMENT_FMIN_TYPE : return INF;
      case ELEMENT_ARGFMIN_TYPE : return INF;
      case ELEMENT_MIN_TYPE : return INF;
      case ELEMENT_ARGMIN_TYPE : return INF;
      default: throw std::runtime_error("Unsupported reduction : no neutral element known");
    }
  }

private:
  virtual std::string generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const & mapping) const = 0;
  virtual int is_invalid_impl(driver::Device const &, expression_tree const &) const;

public:
  base(size_t s, size_t ls0, size_t ls1);
  std::string generate(std::string const & suffix, expression_tree const & tree, driver::Device const & device);
  int is_invalid(expression_tree const & tree, driver::Device const & device) const;
  virtual size_t temporary_workspace(expression_tree const & tree) const;
  virtual size_t lmem_usage(expression_tree const & tree) const;
  virtual size_t registers_usage(expression_tree const & tree) const;
  virtual std::vector<int_t> input_sizes(expression_tree const & tree) const = 0;
  virtual void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix,expression_tree const & tree, runtime::environment const & opt) = 0;

protected:
  size_t simd_width;
  size_t local_size_0;
  size_t local_size_1;
};

}
}

#endif
