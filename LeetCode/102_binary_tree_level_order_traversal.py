"""
102. Binary Tree Level Order Traversal — Medium, ~20 min

Given the root of a binary tree, return the level order traversal of its nodes' values
— i.e., left to right, level by level.

Example: root = [3, 9, 20, null, null, 15, 7] → [[3], [9, 20], [15, 7]]
Example: root = [1] → [[1]]
Example: root = [] → []

Run the checklist out loud, starting from step 1:

1. Restate the problem + clarify edge cases (empty tree? single node? unbalanced tree?)
2. State your approach + complexity — before writing code
3. Code
4. Test on a normal case + an edge case
5. State final time/space complexity
"""

""" 
Example tree: [3, 9, 20, null, null, 15, 7]
3
/ \
9  20
    /\
    15 7
"""

from collections import deque


class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right

root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))


def bfs(root: TreeNode) -> list[list[int]]:
    queue = deque([root])
    result = []

    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)

    return result



# ---

from collections import deque

class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right

    
root = TreeNode(3, TreeNode(9, None, None), TreeNode(20, TreeNode(15, None, None), TreeNode(7, None, None)))


def bfs(root: TreeNode) -> list[list[int]]:
    """
    FIFO
    1. queue [3] -> level [3] -> queue [9, 20]
    """
    queue = deque([root])
    result = []

    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)



# ---

from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))

def level_order(root: TreeNode) -> list[list[int]]:
    queue = deque([root])
    result = []

    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)

        result.append(level)

# ---

from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# example tree: [3, 9, 20, null, null, 15, 7]
root = TreeNode(3,
    TreeNode(9),
    TreeNode(20, TreeNode(15), TreeNode(7)))


def level_order(root: TreeNode) -> list[list[int]]:
    """
    TODO: implement — narrate your approach before coding.
    """
    queue = deque([root])              # BFS queue, seeded with the root node
    result = []                        # holds one list of values per level

    while queue:                       # keep going until every level is consumed
        level_size = len(queue)        # number of nodes currently in this level
        level = []                     # values collected for this level only
        for _ in range(level_size):    # process exactly this level's nodes, not any children just added
            node = queue.popleft()     # dequeue next node in this level (FIFO order)
            level.append(node.val)     # record its value
            if node.left: queue.append(node.left)   # enqueue left child for the next level
            if node.right: queue.append(node.right) # enqueue right child for the next level
        result.append(level)           # this level is done, save it

    return result                      # list of levels, each left-to-right


print(level_order(root))
