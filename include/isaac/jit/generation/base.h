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

namespace runtime
{
  class environment;
}

namespace templates
{

enum fetching_policy_type
{
  FETCH_FROM_LOCAL,
  FETCH_FROM_GLOBAL_STRIDED,
  FETCH_FROM_GLOBAL_CONTIGUOUS
};

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
                              std::string const & idx, std::string const & bound, std::string const & domain_id, std::string const & domain_size,
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

  static void compute_reduce_1d(genstream & os, std::string acc, std::string cur, token const & op);
  static void compute_index_reduce_1d(genstream & os, std::string acc, std::string cur, std::string const & acc_value, std::string const & cur_value, token const & op);
  static std::string neutral_element(token const & op, driver::backend_type backend, std::string const & dtype);

private:
  virtual std::string generate_impl(std::string const & suffix, expression const & tree, driver::Device const & device, symbolic::symbols_table const & mapping) const = 0;
  virtual void check_valid_impl(driver::Device const &, expression const &) const;

public:
  base(size_t s, size_t ls0, size_t ls1);
  std::string generate(std::string const & suffix, expression const & tree, driver::Device const & device);
  void check_valid(expression const & tree, driver::Device const & device) const;
  virtual size_t temporary_workspace(expression const & tree) const;
  virtual size_t lmem_usage(expression const & tree) const;
  virtual size_t registers_usage(expression const & tree) const;
  virtual std::vector<int_t> input_sizes(expression const & tree) const = 0;
  virtual void enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix,expression const & tree, runtime::environment const & opt) = 0;

protected:
  size_t simd_width;
  size_t local_size_0;
  size_t local_size_1;
};

}
}

#endif
