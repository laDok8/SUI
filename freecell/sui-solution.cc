#include "search-strategies.h"
#include "memusage.h"

#include <set>
#include <queue>
#include <stack>
#include <algorithm>

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
    return 0;
}

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state)
{
    return {};
}
