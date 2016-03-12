#include <vector>
#include <map>
#include <set>
#include <functional>
#include <algorithm>
#include <iostream>

namespace isaac
{

typedef float cost_t;
typedef float time_t;
typedef size_t job_t;
typedef size_t proc_t;
typedef std::vector<size_t> procs_t;

typedef std::multimap<job_t, job_t> dag_t;

typedef std::function<cost_t (job_t, proc_t)> compcostfun_t;
typedef std::function<cost_t (job_t, job_t, proc_t, proc_t)> commcostfun_t;

struct order_t
{
    job_t job;
    time_t start;
    time_t end;
};
typedef std::map<proc_t, std::vector<order_t> > orders_t;
typedef std::map<job_t, proc_t> jobson_t;

cost_t wbar(job_t job, procs_t const & procs, compcostfun_t const & compcost)
{
    cost_t res = 0;
    for(proc_t const & a: procs){ res += compcost(job, a); }
    return res/procs.size();
}

cost_t cbar(job_t jobi, job_t jobj, procs_t const & procs, commcostfun_t const & commcost)
{
    cost_t res = 0;
    size_t N = procs.size();
    for(size_t i = 0 ; i < N ; ++i)
        for(size_t j = i + 1 ; j < N ; ++j)
            res += commcost(jobi, jobj, i, j);
    size_t npairs = N*(N-1)/2;
    return res/npairs;
}

cost_t ranku(job_t ni, dag_t const & dag, procs_t const & procs, compcostfun_t const & compcost, commcostfun_t const & commcost)
{
    auto rank = [&](job_t n){ return ranku(n, dag, procs, compcost, commcost); };
    auto w = [&](job_t n){ return wbar(n, procs, compcost); };
    auto c = [&](job_t n1, job_t n2){ return cbar(n1, n2, procs, commcost); };
    if(dag.find(ni)==dag.end())
        return w(ni);
    else
    {
        cost_t res = 0;
        auto lohi = dag.equal_range(ni);
        for(auto it = lohi.first ; it != lohi.second ; ++it){
            job_t nj = it->second;
            res = std::max(res, c(ni, nj) + rank(nj));
        }
        return w(ni) + res;
    }
}

time_t end_time(job_t job, std::vector<order_t> const & events)
{
    for(order_t e: events)
        if(e.job==job)
            return e.end;
    return INFINITY;
}

time_t find_first_gap(typename orders_t::mapped_type const & proc_orders, time_t desired_start, time_t duration)
{
    if(proc_orders.empty())
        return desired_start;
    for(size_t i = 0 ; i < proc_orders.size() ; ++i)
    {
        time_t earliest = std::max(desired_start, (i==0)?0:proc_orders[i-1].end);
        if(proc_orders[i].start - earliest > duration)
            return earliest;
    }
    return std::max(proc_orders.back().end, desired_start);
}

time_t start_time(job_t job, proc_t proc, orders_t const & orders, dag_t const & prec, jobson_t const & jobson,
                  compcostfun_t const & compcost, commcostfun_t const & commcost)
{
    time_t duration = compcost(job, proc);
    time_t comm_ready = 0;
    if(prec.find(job)!=prec.end()){
        auto lohi = prec.equal_range(job);
        for(auto it = lohi.first ; it != lohi.second ; ++it){
            job_t other_job = it->second;
            proc_t other_proc = jobson.at(other_job);
            comm_ready = std::max(comm_ready, end_time(other_job, orders.at(other_proc)) + commcost(other_job, job, proc, other_proc));
        }
    }
    return find_first_gap(orders.at(proc), comm_ready, duration);

}

void allocate(job_t job, orders_t & orders, jobson_t & jobson, dag_t const & prec, compcostfun_t const & compcost, commcostfun_t const & commcost)
{
    auto start = [&](proc_t proc){ return start_time(job, proc, orders, prec, jobson, compcost, commcost);};
    auto finish = [&](proc_t proc) { return start(proc) + compcost(job, proc); };
    proc_t proc = orders.begin()->first;
    for(auto const & pair: orders){
        if(finish(pair.first) < finish(proc))
            proc = pair.first;
    }
    typename orders_t::mapped_type & orders_list = orders[proc];
    orders_list.push_back({job, start(proc), finish(proc)});
    //Update
    std::sort(orders_list.begin(), orders_list.end(), [](order_t const & o1, order_t const & o2){ return o1.start < o2.start;});
    jobson[job] = proc;
}

time_t makespan(orders_t const & orders)
{
    time_t res = 0;
    for(auto const & x: orders)
        res = std::max(res, x.second.back().end);
    return res;
}

void schedule(dag_t const & succ, procs_t const & procs, compcostfun_t const & compcost, commcostfun_t const & commcost, orders_t & orders, jobson_t & jobson)
{
    //Get precedence
    dag_t prec;
    for(auto const & pair: succ)
        prec.insert({pair.second, pair.first});

    //Prioritize jobs
    auto rank = [&](job_t const & job) { return ranku(job, succ, procs, compcost, commcost); };
    auto rank_compare = [&](job_t const & t1, job_t const & t2){ return rank(t1) < rank(t2); };
    std::set<job_t, std::function<bool (job_t, job_t)> > jobs(rank_compare);
    for(auto const & pair: succ){
        jobs.insert(pair.first);
        jobs.insert(pair.second);
    }

    //Assign job to processor
    orders = orders_t();
    jobson = jobson_t();
    for(proc_t proc: procs) orders.insert({proc, {}});
    for(auto it = jobs.rbegin(); it != jobs.rend() ; ++it)
        allocate(*it, orders, jobson, prec, compcost, commcost);
}

}

int main()
{
    isaac::dag_t
          succ = { {1,2}, {1,3}, {1,4}, {1,5}, {1,6},
                   {2,8}, {2,9},
                   {3,7},
                   {4,8}, {4,9},
                   {5,9},
                   {6,8},
                   {7,10},
                   {8,10},
                   {9,10}};

    std::map<size_t, std::vector<size_t> >
         compcosts = {{1, {14,16,9}},
                      {2, {13, 19, 18}},
                      {3, {11, 13, 19}},
                      {4, {13, 8, 17}},
                      {5, {12, 13, 10}},
                      {6, {13, 16, 9}},
                      {7, {7, 15, 11}},
                      {8, {5, 11, 14}},
                      {9, {18, 12, 20}},
                      {10, {21, 7, 16}}};

    std::map<std::pair<size_t, size_t>, size_t>
        commcosts = { {{1,2},18},
                     {{1,3},12},
                     {{1,4},9},
                     {{1,5},11},
                     {{1,6},14},
                     {{2,8},19},
                     {{2,9},16},
                     {{3,7}, 23},
                     {{4,8}, 27},
                     {{4,9}, 23},
                     {{5,9}, 13},
                     {{6,8}, 15},
                     {{7,10}, 17},
                     {{8,10}, 11},
                     {{9,10}, 13} };


    isaac::orders_t orders;
    isaac::jobson_t jobson;

    auto compcost = [&](size_t n, size_t p){ return compcosts[n][p]; };
    auto commcost = [&](size_t n1, size_t n2, size_t p1, size_t p2){ return (p1==p2)?0:commcosts[{n1, n2}]; };

    isaac::schedule(succ, {0,1,2}, compcost, commcost, orders, jobson);

    std::cout << isaac::makespan(orders) << std::endl;
}
