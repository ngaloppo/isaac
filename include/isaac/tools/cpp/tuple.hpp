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

#ifndef ISAAC_TOOLS_CPP_TUPLES_HPP
#define ISAAC_TOOLS_CPP_TUPLES_HPP

#include <vector>
#include <iostream>
#include "isaac/defines.h"
#include "isaac/types.h"

namespace isaac
{

class tuple
{
    friend std::ostream& operator<<(std::ostream & oss, tuple const &);
public:
    tuple() {}
    tuple(std::vector<int_t> const & list): data_(list){}
    tuple(std::initializer_list<int_t> const & list) : data_(list){}
    tuple(int_t a) : data_{a} {}
    tuple(int_t a, int_t b) : data_{a, b} {}

    std::vector<int_t>::iterator begin() { return data_.begin(); }
    std::vector<int_t>::const_iterator begin() const { return data_.begin(); }
    std::vector<int_t>::iterator end() { return data_.end(); }
    std::vector<int_t>::const_iterator end() const { return data_.end(); }

    size_t size() const { return data_.size(); }
    int_t max() const   { return std::accumulate(data_.begin(), data_.end(), std::numeric_limits<int_t>::min(), [](int_t a, int_t b){ return std::max(a, b); }); }
    int_t min() const   { return std::accumulate(data_.begin(), data_.end(), std::numeric_limits<int_t>::max(), [](int_t a, int_t b){ return std::min(a, b); }); }
    int_t prod() const  { return std::accumulate(data_.begin(), data_.end(), 1, std::multiplies<int>()); }

    int_t front() const { return data_.front(); }
    int_t back() const { return data_.back(); }

    int_t& operator[](size_t i) { return data_[i]; }
    int_t operator[](size_t i) const { return data_[i]; }

    operator std::vector<int_t>() const { return data_; }
private:
    std::vector<int_t> data_;
};

inline ISAACAPI std::ostream& operator<<(std::ostream & oss, tuple const &shape)
{
  for(int_t x: shape.data_)
    oss << x << ',';
  return oss;
}

inline ISAACAPI int_t max(tuple const & tp)
{

}

}

#endif
