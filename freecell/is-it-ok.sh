#/bin/bash

if [ "$#" != 1 ]
then
	echo "Usage: $0 SUI_SOLUTION_CC" >&2
	exit 2
fi

OBJFILES='/pub/courses/sui/proj-1/build/*'
SOLUTION_OBJ_FILE='sui-solution.o'

BINARY=sui-proj1-is-it-ok

g++ -I /pub/courses/sui/proj-1/include -std=c++17 -Wall -Wextra -pedantic -O2 -c $1 -o $SOLUTION_OBJ_FILE || exit 1
g++ $OBJFILES $SOLUTION_OBJ_FILE -lpthread -o $BINARY || exit 1

echo -n "BFS: "
./$BINARY --easy-mode 15 --solver bfs 5 1

echo -n "DFS: "
./$BINARY --easy-mode 15 --solver dfs --dls-limit 8 5 2

echo -n "A*n: "
./$BINARY --easy-mode 30 --solver a_star --heuristic nb_not_home 5 3

echo -n "A*s: "
./$BINARY --easy-mode 60 --solver a_star --heuristic student 5 4
