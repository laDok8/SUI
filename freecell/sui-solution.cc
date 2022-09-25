#include <set>
#include <queue>
#include <stack>

#include "search-strategies.h"
#include "memusage.h"

std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state)
{
    std::queue<std::pair<SearchState, std::shared_ptr<std::vector<SearchAction>>>> open;
    std::set<SearchState> closed;

    open.push({init_state, std::make_shared<std::vector<SearchAction>>()});
    while (!open.empty())
    {
        auto node = open.front();
        open.pop();

        if (node.first.isFinal())
            return *node.second;

        if (closed.find(node.first) != closed.end())
            continue;

        closed.insert(node.first);

        auto actions = node.first.actions();
        for (auto action : actions)
        {
            auto nextState = action.execute(node.first);
            auto newPath = std::make_shared<std::vector<SearchAction>>(*node.second);
            newPath->push_back(action);
            open.push({nextState, newPath});
        }
    }

    return {};
}


std::vector<SearchAction> DepthFirstSearch::solve(const SearchState &init_state)
{
    std::stack<std::pair<SearchState, std::shared_ptr<std::vector<SearchAction>>>> open;
    std::set<SearchState> closed;

    open.push({init_state, std::make_shared<std::vector<SearchAction>>()});
    while (!open.empty())
    {
        auto node = open.top();
        open.pop();

        if (node.second->size() > depth_limit_)
            continue;

        if (node.first.isFinal())
            return *node.second;

        if (closed.find(node.first) != closed.end())
            continue;

        closed.insert(node.first);

        auto actions = node.first.actions();
        for (auto action : actions)
        {
            auto nextState = action.execute(node.first);
            auto newPath = std::make_shared<std::vector<SearchAction>>(*node.second);
            newPath->push_back(action);
            open.push({nextState, newPath});
        }
    }

    return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const{
    return 0;
}

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state)
{   
    return {};
}
