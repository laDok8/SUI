#include <set>
#include <queue>
#include "search-strategies.h"
#include "memusage.h"

std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state)
{
    std::queue<std::pair<SearchState, std::vector<SearchAction>>> open;
    std::set<SearchState> closed;

    SearchState working_state(init_state);
    std::pair<SearchState, std::vector<SearchAction>> node(working_state, {});
    open.push(node);

    while (!open.empty())
    {
        auto currentNode = open.front();
        open.pop();

        closed.insert(currentNode.first);

        if (currentNode.first.isFinal())
        {
            return currentNode.second;
        }

        auto nodeActions = currentNode.second;
        auto actions = currentNode.first.actions();

        for (auto action : actions)
        {
            auto appended = nodeActions;
            if (getCurrentRSS() > mem_limit_ - sizeof(SearchAction))
                return {};

            appended.push_back(action);

            auto nextState = action.execute(currentNode.first);
            std::pair<SearchState, std::vector<SearchAction>> nextNode(nextState, appended);
            if (closed.find(nextState) == closed.end())
            {
                if (getCurrentRSS() > mem_limit_ - sizeof(std::pair<SearchState, std::vector<SearchAction>>))
                    return {};

                open.push(nextNode);
            }
        }
    }
    return {};
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState &init_state)
{
    return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const
{
    return 0;
}

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state)
{
    return {};
}
