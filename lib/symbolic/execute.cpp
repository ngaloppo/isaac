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

#include <assert.h>
#include <list>
#include <vector>
#include <stdexcept>
#include "isaac/types.h"
#include "isaac/array.h"
#include "isaac/profiles/profiles.h"
#include "isaac/symbolic/expression/expression.h"
#include "isaac/symbolic/expression/preset.h"
#include "isaac/symbolic/expression/io.h"

namespace isaac
{

namespace symbolic
{

  namespace detail
  {
      typedef std::vector<std::pair<size_t, expression_type> > breakpoints_t;

      inline bool is_elementwise(expression_type type)
      { return type == ELEMENTWISE_1D || type == ELEMENTWISE_2D; }

      /** @brief Optimizes the given expression tree */
//      expression_type optimize(expression_type & tree, size_t idx)
//      {
//        expression_tree::node & node = tree[idx];
//        if(node.type==COMPOSITE_OPERATOR_TYPE)
//        {
//          //Remove useless reshape
//          if(node.binary_operator.op.type==RESHAPE)
//        }
//      }

      /** @brief Parses the breakpoints for a given expression tree */
      expression_type parse(expression_tree const & tree, size_t idx, breakpoints_t & bp)
      {
        expression_tree::node const & node = tree[idx];
        if(node.type==COMPOSITE_OPERATOR_TYPE)
        {
          size_t lidx = node.binary_operator.lhs;
          size_t ridx = node.binary_operator.rhs;
          expression_type ltype = parse(tree, lidx, bp);
          expression_type rtype = parse(tree, ridx, bp);
          op_element const & op = node.binary_operator.op;
          //Reduction
          if(op.type_family==REDUCE || op.type_family==REDUCE_ROWS || op.type_family==REDUCE_COLUMNS)
          {
            if(!is_elementwise(ltype)) bp.push_back({lidx, ltype});
            if(!is_elementwise(rtype)) bp.push_back({ridx, rtype});
            if(op.type_family==REDUCE) return REDUCE_1D;
            if(op.type_family==REDUCE_ROWS) return REDUCE_2D_ROWS;
            if(op.type_family==REDUCE_COLUMNS) return REDUCE_2D_COLS;
          }
          //Matrix Product
          if(op.type_family==MATRIX_PRODUCT)
          {
            if(tree[lidx].type!=DENSE_ARRAY_TYPE) bp.push_back({lidx, ltype});
            if(tree[ridx].type!=DENSE_ARRAY_TYPE) bp.push_back({ridx, rtype});
            if(op.type==MATRIX_PRODUCT_NN_TYPE) return MATRIX_PRODUCT_NN;
            if(op.type==MATRIX_PRODUCT_TN_TYPE) return MATRIX_PRODUCT_TN;
            if(op.type==MATRIX_PRODUCT_NT_TYPE) return MATRIX_PRODUCT_NT;
            if(op.type==MATRIX_PRODUCT_TT_TYPE) return MATRIX_PRODUCT_TT;
          }
          //Arithmetic
          if(op.type_family==UNARY_ARITHMETIC || op.type_family==BINARY_ARITHMETIC)
          {
            //Non-elementwise kernels are temporaries when reshaped
            if(op.type==RESHAPE_TYPE && !is_elementwise(ltype))
              bp.push_back({lidx, ltype});
            else
            {
              //Matrix-Products are temporaries when not assigned
              for(expression_type type: std::vector<expression_type>{MATRIX_PRODUCT_NN,MATRIX_PRODUCT_TN,MATRIX_PRODUCT_NT,MATRIX_PRODUCT_TT})
              {
                if(ltype==type)
                  bp.push_back({lidx, ltype});
                if(rtype==type && op.type!=ASSIGN_TYPE)
                  bp.push_back({ridx, rtype});
                if(rtype==type && op.type==ASSIGN_TYPE)
                  return type;
              }
              //Reductions
              for(expression_type type: std::vector<expression_type>{REDUCE_2D_ROWS, REDUCE_2D_COLS, REDUCE_1D})
              {
                if(ltype==type && ltype==rtype && tree[tree[lidx].binary_operator.lhs].shape == tree[tree[ridx].binary_operator.lhs].shape)
                  return type;
                if(ltype==type && !is_elementwise(rtype))
                  bp.push_back({ridx, rtype});
                if(!is_elementwise(ltype) && rtype==type)
                  bp.push_back({lidx, ltype});
                if((ltype==type && rtype==ELEMENTWISE_1D) || (ltype==ELEMENTWISE_1D && rtype==type))
                  return type;
              }
            }
        }
      }
      if(numgt1(node.shape)<=1)
        return ELEMENTWISE_1D;
      else
        return ELEMENTWISE_2D;
    }
  }

  /** @brief Executes a expression_tree on the given models map*/
  void execute(execution_handler const & c, profiles::map_type & profiles)
  {
    typedef isaac::array array;
    expression_tree tree = c.x();
    /*----Optimize----*/
//    detail::optimize(tree);
    /*----Process-----*/
    driver::Context const & context = tree.context();
    size_t rootidx = tree.root();
    std::vector<std::shared_ptr<array> > temporaries;
    expression_type final_type;
    /*----Matrix Product-----*/
    if(symbolic::preset::matrix_product::args args = symbolic::preset::matrix_product::check(tree.data(), rootidx)){
        final_type = args.type;
    }
    /*----Default-----*/
    else
    {
        expression_tree::node & root = tree[rootidx];
        expression_tree::node & lhs = tree[root.binary_operator.lhs], &rhs = tree[root.binary_operator.rhs];
        expression_tree::node root_save = root, lhs_save = lhs, rhs_save = rhs;

        detail::breakpoints_t breakpoints;
        breakpoints.reserve(8);
        /*----Parse required temporaries-----*/
        final_type = detail::parse(tree, rootidx, breakpoints);
        std::set<size_t> found;
        breakpoints.erase(std::remove_if(breakpoints.begin(), breakpoints.end(), [&](detail::breakpoints_t::value_type const & x){return !found.insert(x.first).second;}), breakpoints.end());
        /*----Compute required temporaries----*/
        for(auto current: breakpoints)
        {
          expression_tree::node const & node = tree[current.first];
          expression_type type = current.second;
          std::shared_ptr<profiles::value_type> const & profile = profiles[std::make_pair(type, node.dtype)];

          //Create temporary
          std::shared_ptr<array> tmp = std::make_shared<array>(node.shape, node.dtype, context);
          temporaries.push_back(tmp);

          //Compute temporary
          root.binary_operator.op.type = ASSIGN_TYPE;
          root.shape = node.shape;
          root.dtype = node.dtype;
          lhs = expression_tree::node(*tmp);
          rhs = node;
          profile->execute(execution_handler(tree, c.execution_options(), c.dispatcher_options(), c.compilation_options()));
          //Update the expression tree
          root = root_save;
          lhs = lhs_save;
          rhs = rhs_save;
          tree[current.first] = expression_tree::node(*tmp);
        }
    }

    /*-----Compute final expression-----*/
    profiles[std::make_pair(final_type, tree[rootidx].dtype)]->execute(execution_handler(tree, c.execution_options(), c.dispatcher_options(), c.compilation_options()));
  }

  void execute(execution_handler const & c)
  {
    execute(c, isaac::profiles::get(c.execution_options().queue(c.x().context())));
  }

}

}
