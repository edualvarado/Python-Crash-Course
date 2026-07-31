"""
Given an m x n 2D grid of '1's (land) and '0's (water), return the number of islands. An island is surrounded by water and formed by connecting adjacent lands horizontally or vertically.

000x000
00xxx00
0000000
00x0000

number_of_islands(grid) = 2
Assume all four edges of the grid are surrounded by water?
Arbitrary m and n (not necessarily square)
"""

"""
- Iterative label relaxation (like connected-component labeling in image processing, or a Game-of-Life-style fixed point): give every land cell its own unique label, then repeatedly scan the whole grid setting label[cell] = min(label[cell], label[neighbors]) until nothing changes in a full pass. Count distinct labels left. Naive because you don't know when to stop — could take O(rows+cols) passes over O(cells) each, so worst case O(cells·(rows+cols)) vs DFS/BFS's O(cells).
"""

def num_islands_dfs(grid: list[list[int]]) -> int:
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if (r, c) in visited or grid[r][c] == 0:
            return

        visited.add((r,c))

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dfs(r + dr, c + dc)

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and (i,j) not in visited:
                counter += 1
                dfs(i, j)



def num_islands(grid: list[list[int]]) -> int:
    """Iterative label relaxation (connected-component labeling)."""
    rows = len(grid)
    cols = len(grid[0])

    # 1. Every LAND cell gets a unique label. Water cells get NO entry
    #    at all -- avoids the label-0 collision from before, since water
    #    never becomes a valid comparison value.
    label = {}
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                label[(i, j)] = i * cols + j

    # 2. Relax until stable: each land cell adopts the smallest label
    #    among itself and its LAND neighbors. Water neighbors are simply
    #    absent from `label`, so the `in label` check excludes them --
    #    they never enter the min() at all.
    changed = True
    while changed:
        changed = False
        for (i, j) in label:
            best = label[(i, j)]
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if (ni, nj) in label:          # land neighbor only
                    best = min(best, label[(ni, nj)])
            if best != label[(i, j)]:
                label[(i, j)] = best
                changed = True

    # 3. Islands = distinct labels remaining among land cells.
    return len(set(label.values()))

"""
Complexity to state out loud:
- Time: each pass is O(cells); worst case (a long snake-shaped island) a label needs O(rows+cols) passes to propagate end to end → O(cells · (rows+cols)).
- Space: O(cells) for the label dict.
- Contrast when you get to DFS/BFS: O(cells) time, single pass, no propagation delay — that's the actual reason this one is "naive."
"""

def num_islands_new(grid: list[list[int]]) -> int:
    """
    v1. 1. Set unique labels for each land piece
    2. Scan/relax labels based on neightbours
    {(0,0): 0, (0,1): 0, (2,1): 8...}
    return set(land.values())
    """

    # Dict containing land coordinates and unique label
    land = {}

    rows = len(grid)
    cols = len(grid[0])

    # Build dict containing unique labels for land only
    # {(0,0): 0, {1,2}: 3, {1,3}: 4}
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                land[(i, j)] = i * cols + j

    # Scan
    changed = True
    while changed:
        changed = False
        for (i,j) in land:
            best = land[(i,j)]
            for (ni, nj) in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                di, dj = i + ni, j + nj
                if (di, dj) in land:
                    best = min(best, land[(di, dj)])
            if best != land[(i,j)]:
                land[(i,j)] = best
                changed = True

    # {(0,0): 0, {1,2}: 3, {1,3}: 3}
    # It would return 2 (from island 0 and island 3)
    return len(set(land.values()))


# Example
grid = [
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1]
]
print(num_islands_new(grid))