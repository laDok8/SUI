#include <set>
#include <queue>
#include "search-strategies.h"


//bool find(std::vector<std::shared_ptr<SearchState>> list, std::shared_ptr<SearchState> element){
//    for( auto e : list){
//        if(element.get() == e.get())
//            return true;
//    }
//    return false;
//}


std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state) {
    std::queue<std::pair<std::shared_ptr<SearchState>,std::vector<SearchAction>>> open;
    std::vector<std::shared_ptr<SearchState>> closed;

    SearchState working_state(init_state);
    std::vector<SearchAction> empty;
    std::shared_ptr<SearchState> stateP(&working_state);
    //TODO: funguje shared ptr?
    std::pair<std::shared_ptr<SearchState>,std::vector<SearchAction>> node (stateP,empty);
    open.push(node);


    while(!open.empty()){
        auto currentNode = open.front();
        open.pop();
        if ( currentNode.first->isFinal()){
            return currentNode.second;
        }

        auto nodeActions = currentNode.second;

        auto actions = working_state.actions();
        for (auto action : actions){
            auto appended = nodeActions;
            appended.push_back(action);
            auto nextState = action.execute(working_state);
            std::shared_ptr<SearchState> nextStateP(&nextState);
            //if(find(closed,nextStateP)){
            //    continue;
            //}

            std::pair<std::shared_ptr<SearchState>,std::vector<SearchAction>> nextNode (nextStateP,appended);

            std::cout << appended.size() << "nesmi se rovnat" << std::endl;
            std::cout << nodeActions.size() << "nesmi se rovnat" << std::endl;
            open.push(nextNode);
            break;
        }
        closed.push_back(currentNode.first);
    }
    return {};




    //for (size_t i = 0; i < nb_attempts_; ++i) {
    //    std::vector<SearchAction> solution;
    //    SearchState working_state(init_state);
//
    //    for (size_t depth = 0; depth < max_depth_ ; ++depth) {
    //        auto actions = working_state.actions();
//
    //        // on a dead end
    //        if (actions.size() == 0)
    //            break; // start over
//
    //        auto action = actions[0];
    //        // actually, pick a random action
    //        std::sample(actions.begin(), actions.end(), &action, 1, rng_);
//
    //        solution.push_back(action);
    //        working_state = action.execute(working_state);
//
    //        if (working_state.isFinal())
    //            return solution;
    //    }
    //}
//
    //return {};
	return {};
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState &init_state) {
	return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const {
    return 0;
}

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state) {
	return {};
}
