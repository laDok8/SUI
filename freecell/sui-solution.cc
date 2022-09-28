#include "search-strategies.h"
#include "memusage.h"

#include <set>
#include <queue>
#include <stack>
#include <algorithm>
#include <chrono>

#define SEARCH_STATE_SIZE sizeof(SearchState)
#define SEARCH_ACTION_SIZE sizeof(SearchAction)
#define SEARCH_STATE_PTR_SIZE sizeof(SearchState) + sizeof(std::shared_ptr<SearchState>)
#define NODE_SIZE sizeof(std::pair<std::shared_ptr<SearchState>, SearchAction>)

#define CHECK_MEMORY_LIMIT(mem_limit_, alloc_size)  \
    if (getCurrentRSS() + alloc_size >= mem_limit_) \
        return {};

std::vector<SearchAction> reconstructPath(const std::map<std::shared_ptr<SearchState>, std::pair<std::shared_ptr<SearchState>, SearchAction>> &nodes,
                                          const std::shared_ptr<SearchState> &current, const size_t mem_limit_)
{

    std::vector<SearchAction> solution;
    std::shared_ptr<SearchState> state = current;
    while (nodes.find(state) != nodes.end())
    {
        solution.push_back(nodes.at(state).second);
        state = nodes.at(state).first;
    }
    std::reverse(solution.begin(), solution.end());
    return solution;
}

std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state)
{

    std::queue<std::shared_ptr<SearchState>> open;
    std::set<SearchState> closed;
    std::map<std::shared_ptr<SearchState>, std::pair<std::shared_ptr<SearchState>, SearchAction>> nodes;

    if (init_state.isFinal())
        return {};

    open.push(std::make_shared<SearchState>(init_state));

    while (!open.empty())
    {
        auto current = open.front();
        open.pop();

        if (closed.find(*current) != closed.end())
            continue;

        closed.insert(*current);

        for (auto &a : current->actions())
        {
            CHECK_MEMORY_LIMIT(mem_limit_, SEARCH_STATE_PTR_SIZE);
            auto next = std::make_shared<SearchState>(a.execute(*current));
            if (closed.find(*next) != closed.end())
                continue;

            CHECK_MEMORY_LIMIT(mem_limit_, SEARCH_STATE_PTR_SIZE);
            open.push(next);
            CHECK_MEMORY_LIMIT(mem_limit_, NODE_SIZE);
            nodes.insert({next, {current, a}});
            if (next->isFinal())
            {
                return reconstructPath(nodes, next, mem_limit_);
            }
        }
    }

    return {};
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState &init_state)
{
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const
{
    auto homes = 0;
    auto moves = 0;
    for (const auto &home : state.homes)
    {
        auto opt_top = home.topCard();
        if (opt_top.has_value())
            homes += opt_top->value;
    }

    for (const auto &stack : state.stacks)
    {
        moves += std::pow(stack.nbCards(), 2);
    }

    return 52 - homes + moves;
}

struct Node
{
    std::shared_ptr<SearchState> state;
    double cost;
};

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state)
{
    std::priority_queue<std::pair<double, std::shared_ptr<SearchState>>, std::vector<std::pair<double, std::shared_ptr<SearchState>>>, std::greater<std::pair<double, std::shared_ptr<SearchState>>>> open;
    std::set<SearchState> closed;
    std::map<std::shared_ptr<SearchState>, Node> nodes;
    std::map<std::shared_ptr<SearchState>, SearchAction> actions;

    if (init_state.isFinal())
        return {};

    auto init = std::make_shared<SearchState>(init_state);
    open.push({0, init});
    nodes.insert({init, {init, 0}});

    auto t0 = std::chrono::steady_clock::now();
    while (!open.empty())
    {
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count() > 15)
            return {};

        auto current = open.top().second;
        open.pop();

        if (closed.find(*current) != closed.end())
            continue;

        closed.insert(*current);

        for (auto &a : current->actions())
        {
            auto next = std::make_shared<SearchState>(a.execute(*current));
            if (closed.find(*next) != closed.end())
                continue;

            double cost = nodes.at(current).cost + 1;
            if (nodes.find(next) == nodes.end())
            {
                nodes.insert({next, {current, cost}});
                actions.insert({next, a});
                open.push({cost + compute_heuristic(*next, *heuristic_), next});
                if (next->isFinal())
                {
                    std::vector<SearchAction> solution;
                    while (actions.find(next) != actions.end())
                    {
                        solution.push_back(actions.at(next));
                        next = nodes.at(next).state;
                    }
                    std::reverse(solution.begin(), solution.end());
                    return solution;
                }
            }
        }
    }

    return {};
}