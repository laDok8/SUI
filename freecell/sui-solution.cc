#include "search-strategies.h"
#include "memusage.h"

#include <set>
#include <queue>
#include <stack>
#include <algorithm>

std::vector<SearchAction> reconstructPath(const std::map<std::shared_ptr<SearchState>, std::pair<std::shared_ptr<SearchState>, SearchAction>>& parent,
                                          const std::shared_ptr<SearchState>& current)
{
    std::vector<SearchAction> solution;
    std::shared_ptr<SearchState> state = current;
    while (parent.find(state) != parent.end())
    {
        solution.push_back(parent.at(state).second);
        state = parent.at(state).first;
    }
    std::reverse(solution.begin(), solution.end());
    return solution;
}

std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state)
{
    std::queue<std::shared_ptr<SearchState>> open;
    std::set<SearchState> closed;
    std::map<std::shared_ptr<SearchState>, std::pair<std::shared_ptr<SearchState>, SearchAction>> parent;

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
            auto next = std::make_shared<SearchState>(a.execute(*current));
            if (closed.find(*next) != closed.end())
                continue;

            open.push(next);
            parent.insert({next, {current, a}});
            if (next->isFinal()) {
                //std::cerr << "current: " << getCurrentRSS() << " peak: " << getPeakRSS() << std::endl;
                return reconstructPath(parent, next);
            }
        }
    }
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
