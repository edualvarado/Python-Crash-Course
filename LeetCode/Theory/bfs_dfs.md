# BFS & DFS on trees and graphs

BFS and DFS both visit every node in a tree/graph exactly once — they differ in the order and the data structure driving that order.

BFS — queue (FIFO), explores level by level, outward from the start node. Reach for it when you need the shortest path / fewest steps in an unweighted graph, or when the problem itself is level-shaped (level order traversal, "minimum number of moves").

```python
from collections import deque

queue = deque([root])
result = []
while queue:
    level_size = len(queue)
    level = []
    for _ in range(level_size):
        node = queue.popleft()
        level.append(node.val)
        if node.left:  queue.append(node.left)
        if node.right: queue.append(node.right)
    result.append(level)
```

DFS — stack (explicit or via recursion), explores one branch all the way down before backtracking. Reach for it when you need to explore full connectivity, count/label components, check existence of a path, or when the problem recurses naturally on subtree/substructure (path sum, validate BST, backtracking).

```python
def dfs(node, visited):
    if node in visited:
        return
    visited.add(node)
    # process node
    for neighbor in neighbors(node):
        dfs(neighbor, visited)
```

On a grid (treat each cell as a node, adjacency = up/down/left/right), DFS/BFS becomes flood fill — the standard way to count connected components:

```python
def dfs(r, c):
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return
    if (r, c) in visited or grid[r][c] == '0':
        return
    visited.add((r, c))
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        dfs(r + dr, c + dc)
```

One-liner for the interviewer: "BFS explores level by level with a queue — I reach for it when I need shortest path or a level-grouped result; DFS explores depth-first with recursion or a stack — I reach for it when I need full connectivity, component counting, or the problem's structure recurses naturally."

Maps directly onto today's two problems: #102 (Binary Tree Level Order Traversal) is BFS by definition — the level-by-level grouping is the whole point. #200 (Number of Islands) is DFS/BFS flood fill on a grid — each unvisited '1' starts a new flood fill that marks one connected component (island); the answer is the number of times you start a fresh fill.
