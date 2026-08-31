"""
98. Validate Binary Search Tree — Medium, ~20 min

Given the root of a binary tree, determine if it is a valid BST (binary search tree).

A valid BST means:
- every node in the LEFT subtree of a node has a value strictly LESS than the node's value
- every node in the RIGHT subtree has a value strictly GREATER
- both subtrees are themselves valid BSTs

Note "subtree", not "child" — that distinction is the whole problem.

Input:  [2,1,3]                 ->  True
Input:  [5,1,4,null,null,3,6]   ->  False
Input:  [2,2,2]                 ->  False   (strictly less/greater: no duplicates)

Narrate before coding:
- Draw [5,1,4,null,null,3,6]. Every parent/child pair is locally fine
  (1 < 5, 4 > 5? no — 4 < 5 already breaks it). Now draw [5,1,6,null,null,3,7]:
  every parent/child pair IS locally fine, yet it's not a BST. Say out loud why.
  That's the trap this problem tests — the naive "check node vs its two children"
  answer fails here, and interviewers pick this problem precisely to catch it.
- What information does a node need from its ancestors to know whether it's legal?
- Second angle worth stating: what does an in-order traversal of a valid BST produce?
  That property gives you a completely different solution.
- Complexity to aim for: O(n) time, O(h) space (h = tree height, O(n) worst case
  for a degenerate/skewed tree).

Run the checklist out loud, starting from step 1:

1. Restate the problem + clarify edge cases (empty tree? single node? duplicates?
   value range — can a node hold the minimum 32-bit int?)
2. State your approach + complexity — before writing code
3. Code
4. Test on a normal case + an edge case
5. State final time/space complexity
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# example tree: [5, 1, 4, null, null, 3, 6]  -> NOT a valid BST
#
#         5
#        / \
#       1   4
#          / \
#         3   6

from collections import deque

def is_valid_bst_bounds(root: TreeNode | None) -> bool:
    """
    Recursive with bounds: carry the (low, high) interval a node's value must
    fall inside, tightening it as you descend.

    Time:  O(n)  -- each node visited once
    Space: O(h)  -- recursion stack, h = height (O(n) if the tree is a chain)

    Hint on the interval, not the code: the root can be anything, so it starts
    unbounded. Use float('-inf') / float('inf') rather than hardcoding
    -2**31 / 2**31-1 — a node is allowed to hold exactly the minimum 32-bit int,
    and hardcoding the bound would wrongly reject it.
    """
    # TODO

    """
    
        5
       / \
      1   6
         / \
        3   7

    Iterations:
    1) 
    level_size = 1
    level = []
    node.val = 5
    if node.left: (1)
        if 1 < 5 (yes)
            min_left = 1
            queue [1]
    if node.right: (6)
        if 6 > 5 (yes)
            max_right = 6
            queue [1, 6]
    level = [[5]]

    2) 
    level_size = 2
    level = [[5]]
    node.val = 1
    if node.left: None
    if node.right: None
    level = [[5], [1]]

    node.val = 6
    if node.left: (3)
        if 3 < 6 (yes) and 3 > 5 (no)
            return False

    
    Alt: what if you end up with [[3], [9, 20], [None, None], [15, 7]]

    when, you just need to check if the first element of the list
    """
    if root is None:
        return True

    queue = deque([(root, float('-inf'), float('inf'))])

    while queue:
        node, low, high = queue.popleft()

        if node.val <= low or node.val >= high:
            return False

        if node.left: 
            queue.append((node.left, low, node.val))

        if node.right:
            queue.append((node.right, node.val, high))

    return True


def is_valid_bst_inorder(root: TreeNode | None) -> bool:
    """
    In-order traversal: left -> node -> right.

    Time:  O(n)
    Space: O(h)

    A valid BST's in-order traversal is strictly increasing. You do NOT need to
    materialize the whole list to check that — one variable is enough. Say out
    loud what that variable holds and why "strictly" matters here.

    Recursive or iterative (explicit stack) both count; the iterative version is
    a good thing to be able to write, since it's the same stack pattern as an
    iterative DFS.

                 10
              /      \
            5          15
           / \        /   \
          2   7     12     20
         / \   \      \
        1   3   10     13

f(node) = f(node.left) + [node.val] + f(node.right), with f(None) = []

f(10) = f(5) + [10] + f(15)

  f(5)  = f(2) + [5] + f(7)
    f(2)  = f(1) + [2] + f(3)  =  [] + [1] + []  +  [2]  +  [] + [3] + []  =  [1,2,3]
    f(7)  = f(None) + [7] + f(10)  =  [] + [7] + [10]                      =  [7,10]
  f(5)  = [1,2,3] + [5] + [7,10]                                           =  [1,2,3,5,7,10]

  f(15) = f(12) + [15] + f(20)
    f(12) = f(None) + [12] + f(13)  =  [] + [12] + [13]                    =  [12,13]
    f(20) = []  + [20] + []                                                =  [20]
  f(15) = [12,13] + [15] + [20]                                            =  [12,13,15,20]

f(10) = [1,2,3,5,7,10] + [10] + [12,13,15,20]
      =  1, 2, 3, 5, 7, 10, 10, 12, 13, 15, 20

                 f(10)
              /      \
            5          15
           / \        /   \
          2   7     12     20
         / \   \      \
        1   3   10     13

DFS
- pre-order: node → left → right
- in-order: left → node → right
- post-order: left → right → node

    def dfs(node, visited):
        if node in visited:
            return

        visited.add(node)

        for n in neighbors(node):
            dfs(n, visited)
    """
    prev = float('-inf')

    def inorder(node) -> bool:
        nonlocal prev

        if node is None:
            return True

        # left
        if not inorder(node.left):
            return False

        # node
        if node.val > prev:
            prev = node.val
        else:
            return False

        # right
        return inorder(node.right)

    return inorder(root)
        





# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def build_tree(values: list[int | None]) -> TreeNode | None:
    """
    Build a tree from LeetCode's level-order list form, where None = missing node.
    Test-harness plumbing only — not part of the exercise.
    """
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values):
            v = values[i]
            i += 1
            if v is not None:
                node.left = TreeNode(v)
                queue.append(node.left)
        if i < len(values):
            v = values[i]
            i += 1
            if v is not None:
                node.right = TreeNode(v)
                queue.append(node.right)
    return root


INT_MIN = -2 ** 31
INT_MAX = 2 ** 31 - 1

CASES: list[tuple[str, list[int | None], bool]] = [
    ("normal valid",     [2, 1, 3],                              True),
    ("normal invalid",   [5, 1, 4, None, None, 3, 6],            False),
    ("locally ok trap",  [5, 1, 6, None, None, 3, 7],            False),  # every parent/child pair passes
    ("deep violation",   [10, 5, 15, None, None, 6, 20],         False),  # 6 < 10 but sits in the right subtree
    ("left chain",       [3, 2, None, 1],                        True),
    ("right chain",      [1, None, 2, None, 3],                  True),
    ("dup at root",      [2, 2, 2],                              False),  # strict inequality
    ("dup left child",   [1, 1],                                 False),
    ("single node",      [1],                                    True),
    ("empty",            [],                                     True),
    ("int min value",    [INT_MIN],                              True),   # kills a hardcoded -2**31 bound
    ("int extremes",     [0, INT_MIN, INT_MAX],                  True),
]

SOLVERS = [
    ("bounds",  is_valid_bst_bounds),
    ("inorder", is_valid_bst_inorder),
]

print("98. Validate Binary Search Tree")
for label, solver in SOLVERS:
    print(f"\n  {label}")
    for name, values, expected in CASES:
        got = solver(build_tree(values))
        if got is None:
            print(f"    {name:16s}: not implemented")
            continue
        mark = "OK  " if got == expected else "FAIL"
        print(f"    {name:16s}: {mark} got={got} expected={expected}")
