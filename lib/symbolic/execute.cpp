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

namespace isaac
{

namespace symbolic
{

  namespace detail
  {
    typedef std::vector<std::pair<expression_type, expression_tree::node*> > breakpoints_t;


    inline bool is_mmprod(expression_type x)
    {
        return x==MATRIX_PRODUCT_NN || x==MATRIX_PRODUCT_NT ||
               x==MATRIX_PRODUCT_TN || x==MATRIX_PRODUCT_TT;
    }

    inline bool is_mvprod(expression_type x)
    {
        return x==REDUCE_2D_ROWS || x==REDUCE_2D_COLS;
    }

    inline bool has_temporary(op_element op, expression_type expression, expression_type other, bool is_first)
    {
        bool result = false;
        switch(op.type_family)
        {
            case UNARY:
            case BINARY:
                result |= is_mmprod(expression)
                          || (result |= expression==REDUCE_2D_ROWS && other==REDUCE_2D_COLS)
                          || (result |= expression==REDUCE_2D_COLS && other==REDUCE_2D_ROWS);
                break;
            case REDUCE:
                result |= is_mvprod(expression)
                          || expression==REDUCE_1D;
                break;
            case REDUCE_ROWS:
                result |= is_mmprod(expression)
                          || is_mvprod(expression)
                          || expression==REDUCE_1D;
                break;
            case REDUCE_COLUMNS:
                result |= is_mmprod(expression)
                          || is_mvprod(expression)
                          || expression==REDUCE_1D;
                break;
            case MATRIX_PRODUCT:
                result |= (is_mmprod(expression) && !is_first)
                          || is_mvprod(expression)
                          || expression==REDUCE_1D;
                break;
            default:
                break;
        }
        return result;
    }

    inline expression_type merge(op_element op, expression_type left, expression_type right)
    {
        switch(op.type_family)
        {
            case UNARY:
                if(is_mmprod(left))
                    return ELEMENTWISE_2D;
                return left;
            case BINARY:
                if(left == REDUCE_2D_ROWS || right == REDUCE_2D_ROWS) return REDUCE_2D_ROWS;
                else if(left == REDUCE_2D_COLS || right == REDUCE_2D_COLS) return REDUCE_2D_COLS;
                else if(left == REDUCE_1D || right == REDUCE_1D) return REDUCE_1D;
                else if(left == ELEMENTWISE_2D || right == ELEMENTWISE_2D) return ELEMENTWISE_2D;
                else if(left == ELEMENTWISE_1D || right == ELEMENTWISE_1D) return op.type==OUTER_PROD_TYPE?ELEMENTWISE_2D:ELEMENTWISE_1D;
                else if(is_mmprod(left) || is_mmprod(right)) return ELEMENTWISE_2D;
                else if(right == INVALID_EXPRESSION_TYPE) return left;
                else if(left == INVALID_EXPRESSION_TYPE) return right;
                throw;
            case REDUCE:
                return REDUCE_1D;
            case REDUCE_ROWS:
                return REDUCE_2D_ROWS;
            case REDUCE_COLUMNS:
                return REDUCE_2D_COLS;
            case MATRIX_PRODUCT:
                if(op.type==MATRIX_PRODUCT_NN_TYPE) return MATRIX_PRODUCT_NN;
                else if(op.type==MATRIX_PRODUCT_TN_TYPE) return MATRIX_PRODUCT_TN;
                else if(op.type==MATRIX_PRODUCT_NT_TYPE) return MATRIX_PRODUCT_NT;
                else return MATRIX_PRODUCT_TT;
            default:
                throw;
        }
    }

    /** @brief Parses the breakpoints for a given expression tree */
    expression_type parse(expression_tree&tree, size_t idx, breakpoints_t & breakpoints, bool is_first = true)
    {
      expression_tree::node & node = tree[idx];
      if (node.type == COMPOSITE_OPERATOR_TYPE)
      {
          expression_type type_left = parse(tree, node.binary_operator.lhs, breakpoints, false);
          expression_type type_right = parse(tree, node.binary_operator.rhs, breakpoints, false);
          expression_type result = merge(node.binary_operator.op, type_left, type_right);
          if(has_temporary(node.binary_operator.op, type_left, type_right, is_first))
            breakpoints.push_back({result, &node});
          return result;
      }
      else if(node.type == DENSE_ARRAY_TYPE)
      {
          if(numgt1(node.shape)<=1)
            return ELEMENTWISE_1D;
          else
            return ELEMENTWISE_2D;
      }
      else
        return INVALID_EXPRESSION_TYPE;
    }
  }

  /** @brief Executes a expression_tree on the given models map*/
  void execute(execution_handler const & c, profiles::map_type & profiles)
  {
    typedef isaac::array array;

    expression_tree tree = c.x();
    driver::Context const & context = tree.context();
    size_t rootidx = tree.root();
    std::vector<std::shared_ptr<array> > temporaries;

    expression_type final_type;
    //MATRIX_PRODUCT
    if(symbolic::preset::matrix_product::args args = symbolic::preset::matrix_product::check(tree.data(), rootidx)){
        final_type = args.type;
    }
    //Default
    else
    {
        expression_tree::node & root = tree[rootidx];
        expression_tree::node & lhs = tree[root.binary_operator.lhs], &rhs = tree[root.binary_operator.rhs];
        expression_tree::node root_save = root, lhs_save = lhs, rhs_save = rhs;

        detail::breakpoints_t breakpoints;
        breakpoints.reserve(8);

        //Init
        expression_type current_type;
        if(numgt1(tree.shape())<=1)
          current_type=ELEMENTWISE_1D;
        else
          current_type=ELEMENTWISE_2D;
        final_type = current_type;

        /*----Parse required temporaries-----*/
        final_type = detail::parse(tree, rootidx, breakpoints);

        /*----Compute required temporaries----*/
        for(detail::breakpoints_t::iterator it = breakpoints.begin() ; it != breakpoints.end() ; ++it)
        {
          expression_tree::node const & node = *it->second;
          expression_type type = it->first;
          std::shared_ptr<profiles::value_type> const & profile = profiles[std::make_pair(type, node.dtype)];

          //Create temporary
          std::shared_ptr<array> tmp = std::make_shared<array>(node.shape, node.dtype, context);
          temporaries.push_back(tmp);

          //Compute temporary
          root.binary_operator.op.type = ASSIGN_TYPE;
          expression_tree::fill(lhs, (array&)*tmp);
          rhs = node;
          profile->execute(execution_handler(tree, c.execution_options(), c.dispatcher_options(), c.compilation_options()));

          //Update the expression tree
          root = root_save;
          lhs = lhs_save;
          expression_tree::fill(rhs, (array&)*tmp);
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
