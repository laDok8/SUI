#include "search-strategies.h"
#include "memusage.h"

#include <set>
#include <queue>
#include <stack>
#include <algorithm>

#define MEMORY_MAX 0.95
#define CHECK_MEMORY_LIMIT(mem_limit_)  \
    if (getCurrentRSS() >= mem_limit_) \
        return {}

struct ParentNode
{
    std::shared_ptr<SearchState> state;
    SearchAction action;
};

std::vector<SearchAction> reconstructPath(const std::map<std::shared_ptr<SearchState>, ParentNode> &nodes, std::shared_ptr<SearchState> &current){
    std::vector<SearchAction> solution;
    while (nodes.find(current) != nodes.end())
    {
        solution.push_back(nodes.at(current).action);
        current = nodes.at(current).state;
    }
    std::reverse(solution.begin(), solution.end());
    return solution;
}

std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state)
{
    double memory_limit = mem_limit_ * MEMORY_MAX;
    std::queue<std::shared_ptr<SearchState>> open;
    std::set<SearchState> closed;
    std::map<std::shared_ptr<SearchState>, ParentNode> nodes;

    if (init_state.isFinal())
        return {};

    open.push(std::make_shared<SearchState>(init_state));
    while (!open.empty())
    {
        std::shared_ptr<SearchState> current = open.front();
        open.pop();

        if (closed.find(*current) != closed.end())
            continue;

        closed.insert(*current);
        for (SearchAction &a : current->actions())
        {
            CHECK_MEMORY_LIMIT(memory_limit);
            std::shared_ptr<SearchState> next = std::make_shared<SearchState>(a.execute(*current));
            if (closed.find(*next) != closed.end())
                continue;

            open.push(next);
            nodes.insert({next, {current, a}});
            if (next->isFinal())
                return reconstructPath(nodes, next);
        }
    }
    return {};
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState &init_state)
{
    double memory_limit = mem_limit_ * MEMORY_MAX;
    std::stack<std::pair<int, std::shared_ptr<SearchState>>> open;
    std::set<SearchState> closed;
    std::map<std::shared_ptr<SearchState>, ParentNode> nodes;

    if (init_state.isFinal())
        return {};

    open.push({0, std::make_shared<SearchState>(init_state)});
    while (!open.empty())
    {
        std::shared_ptr<SearchState> current = open.top().second;
        int depth = open.top().first;
        open.pop();

        if (depth >= depth_limit_ || closed.find(*current) != closed.end())
            continue;

        closed.insert(*current);
        for (SearchAction &a : current->actions())
        {
            CHECK_MEMORY_LIMIT(memory_limit);
            std::shared_ptr<SearchState> next = std::make_shared<SearchState>(a.execute(*current));
            if (closed.find(*next) != closed.end())
                continue;

            open.push({depth + 1, next});
            nodes.insert({next, {current, a}});
            if (next->isFinal())
                return reconstructPath(nodes, next);
        }
    }
    return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const
{
    int homes = 0;
    int moves = 0;
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
    double cost;
    double cost_from_start;
    std::shared_ptr<SearchState> state;

    bool operator<(const Node &other) const
    {
        return cost > other.cost;
    }
};

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state)
{
    double memory_limit = mem_limit_ * MEMORY_MAX;
    std::priority_queue<Node> open;
    std::set<SearchState> closed;
    std::map<std::shared_ptr<SearchState>, ParentNode> nodes;

    if (init_state.isFinal())
        return {};

    std::shared_ptr<SearchState> init = std::make_shared<SearchState>(init_state);
    open.push({0, 0, init});

    while (!open.empty())
    {
        std::shared_ptr<SearchState> current = open.top().state;
        double cost_from_start = open.top().cost_from_start;
        open.pop();

        if (closed.find(*current) != closed.end())
            continue;

        closed.insert(*current);
        for (SearchAction &a : current->actions())
        {
            CHECK_MEMORY_LIMIT(memory_limit);
            std::shared_ptr<SearchState> next = std::make_shared<SearchState>(a.execute(*current));
            if (closed.find(*next) != closed.end())
                continue;

            auto next_cost = cost_from_start + 1;

            nodes.insert({next, {current, a}});
            open.push({next_cost + compute_heuristic(*next, *heuristic_), next_cost, next});
            if (next->isFinal())
                return reconstructPath(nodes, next);
        }
    }
    return {};
}