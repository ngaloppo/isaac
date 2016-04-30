#include <set>
#include <fstream>

#include "isaac/array.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/expression.h"
#include "isaac/runtime/scheduler/dag.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{
namespace runtime
{
namespace scheduler
{

int extended_gcd(int a, int b, int &x, int &y)
{
    if (a % b == 0){
        x = 0;
        y = 1;
        return b;
    }

    int newx, newy;
    int ret = extended_gcd(b, a % b, newx, newy);

    x = newy;
    y = newx - newy * (a / b);
    return ret;
}

int gcd(int a, int b)
{
  while(true)
  {
    if (a == 0)
      return b;
    b %= a;
    if (b == 0)
      return a;
    a %= b;
  }
}

int diophantine(int a, int b, int c, int &x, int &y)
{
  int absa = std::abs(a);
  int absb = std::abs(b);
  int absc = std::abs(c);
  int d = gcd(absa, absb);
  if(absc % d > 0)
    return -1;
  int k = absc/d;
  int res = extended_gcd(absa/d, absb/d, x, y);
  x*=k;
  y*=k;
  if(a < 0) x*=-1;
  if(b < 0) y*=-1;
  if(c < 0){
    x *= -1;
    y *= -1;
  }
  return res;
}

int lcm(int a, int b)
{
  int temp = gcd(a, b);
  return temp ? (a / temp * b) : 0;
}

int first_overlap(int starta, int inca, int startb, int incb, int& x, int& y)
{
  if(diophantine(inca, -incb, startb - starta, x, y)==-1)
    return -1;
  //Projects back the val so that it is immediately
  //greater than max(starta, startb) up to lcm
  int val = starta + inca*x;
  int off = val - std::max(starta, startb);
  int l = lcm(inca, incb);
  if(off < 0 || off > l)
    val += (1 - off/l)*l;
  x = (val - starta)/inca;
  y = (val - startb)/incb;
  return val;
}


tuple dag::repack(int_t start, const tuple &ld)
{
  std::vector<int_t> result(ld.size());
  for(size_t i = result.size() - 1 ; i >= 1 ; i--){
    int_t stride = ld[i-1];
    result[i] = start/stride;
    start -= result[i]*stride;
  }
  result[0] = start;
  return result;
}

bool dag::overlap(expression::node const & x, expression::node const & y)
{
  if(x.array.handle.cl != y.array.handle.cl ||
     x.array.handle.cu != y.array.handle.cu ||
     x.shape.size() != y.shape.size())
    return false;
  tuple offx = repack(x.array.start, x.ld);
  tuple offy = repack(y.array.start, y.ld);
  for(size_t i = 0 ; i < x.shape.size() ; ++i){
    int indx, indy;
    if(first_overlap(offx[i], x.ld[i], offy[i], y.ld[i], indx, indy)==-1)
      return false;
    if(indx < 0 || indx >= x.shape[i] || indy < 0 || indy >= y.shape[i])
      return false;
  }
  return true;
}

dag::dag()
{ jobs_.reserve(16); }

array_base& dag::create_temporary(array_base* tmp)
{
  tmp_.push_back(std::shared_ptr<array_base>(tmp));
  return *tmp;
}

void dag::append(expression const & job, std::string const & name)
{
  //Add new job
  jobs_.push_back(job);
  job_t newid = jobs_.size() - 1;
  adjacency_.insert({newid, {}});
  names_.push_back(name.empty()?tools::to_string(newid):name);
  //Add dependencies;
  std::vector<expression::node const *> nread;
  auto extractor = [&](size_t idx, size_t) { if(job[idx].type==DENSE_ARRAY_TYPE) nread.push_back(&job[idx]);};
  traverse_dfs(job, job[job.root()].binary_operator.rhs, extractor);
  for(job_t previd = 0 ; previd < jobs_.size() ; ++previd){
    expression const & current = jobs_[previd];
    expression::node const & assignment = current[current.root()];
    expression::node const & pwritten = current[assignment.binary_operator.lhs];
    auto pred = [&](expression::node const * x){ return overlap(*x, pwritten);};
    bool has_raw = std::find_if(nread.begin(), nread.end(), pred) != nread.end();
    if(has_raw)
      adjacency_[previd].push_back(newid);
  }

}

void dag::export_graphviz(std::string const & path)
{
  std::ofstream out(path);
  out << "digraph {" << std::endl;
  for(auto const & x: adjacency_)
    out << x.first << "[label=\"" << names_[x.first] << "\"];" << std::endl;
  for(auto const & x: adjacency_)
  {
    if(x.second.empty())
      out << x.first << std::endl;
    else
      for(job_t y: x.second)
        out << x.first << " -> " << y << ";" << std::endl;
  }
  out << "}" << std::endl;
}

}
}
}
