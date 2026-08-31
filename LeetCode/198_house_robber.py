"""
198. House Robber — Medium, ~20 min

Houses in a row, each with some amount of money. You can't rob two adjacent
houses (adjacent alarms are linked and call the police). Return the maximum
amount you can rob.

Input:  nums = [1,2,3,1]     ->  4     (rob house 0 and house 2:  1 + 3)
Input:  nums = [2,7,9,3,1]   ->  12    (rob houses 0, 2, 4:  2 + 9 + 1)
Input:  nums = [2,1,1,2]     ->  4     (rob houses 0 and 3:  2 + 2)

Narrate before coding:
- Kill the greedy idea out loud first. "Rob every other house" — try it on
  [2,1,1,2]: houses 0,2 gives 3, houses 1,3 gives 3, but the answer is 4
  (houses 0 and 3). The winning choice SKIPS TWO in a row. Say why that means
  no fixed pattern works, and you have to actually search.
- Now the recurrence. Stand at house i and ask: do I rob it or not?
      - if you rob it   -> you cannot have robbed i-1, so you add nums[i] to the
                           best total achievable up to house i-2
      - if you skip it  -> your total is just the best achievable up to i-1
  Those are the only two options, and you want the better one. Write f(i).
- Compare that to #70 Climbing Stairs. Same two-branch shape, same "look back
  two" structure — but climbing stairs ADDS the branches (counting: either/or
  with no overlap) while this one takes the MAX (optimizing: pick the better).
  Being able to state that difference is worth a lot.
- Base cases: f(0) and f(1). Defend both. What is the answer for an empty list?
- The four functions are the same ladder as #70. Before writing each, say which
  cost you're removing:
      brute_force  ->  memo         : stop recomputing the same subproblem
      memo         ->  tabulation   : stop paying for the call stack
      tabulation   ->  two_vars     : stop storing values you'll never read again

Run the checklist out loud, starting from step 1:

1. Restate the problem + clarify edge cases (empty? one house? zeros? negatives?)
2. State your approach + complexity — before writing code
3. Code
4. Test on a normal case + an edge case
5. State final time/space complexity
"""


def rob_brute_force(nums: list[int]) -> int:
    """
    Plain recursion, no cache. Explore both choices at every house.

    Time:  O(2^n)
    Space: O(n)  -- recursion stack

    Write the recurrence straight down as code. The harness refuses to run this
    past 20 houses — be able to say why.

    [2,7,9,3,1]

    """

    def f(i: int) -> int:
        if i < 0:
            return 0
        return max(f(i - 1), f(i - 2) + nums[i])

    return f(len(nums) - 1)

def rob_memo(nums: list[int]) -> int:
    """
    Top-down DP (memoization): the same recursion, plus a cache.

    Time:  O(n)
    Space: O(n)  -- cache, plus O(n) recursion stack

    Two extra lines over brute force: check the cache before working, write to
    it before returning. Remember from #70 that a dict you only ever mutate
    (cache[k] = ...) does NOT need `nonlocal`.
    """
    cache = {}

    def f(i: int) -> int:
        if i < 0:
            return 0

        if i in cache:
            return cache[i]

        cache[i] = max(f(i - 1), f(i - 2) + nums[i])
        return cache[i]
    
    return f(len(nums) - 1)

def rob_tabulation(nums: list[int]) -> int:
    """
    Bottom-up DP (tabulation): no recursion, fill a table with a loop.

    Time:  O(n)
    Space: O(n)  -- the table

    table[i] should mean one specific thing. Write that sentence down before you
    write the loop — it's the whole trick, and it's what an interviewer will ask
    you to state.

    Mind the bounds: check by hand that a 1-element list doesn't index off the end.
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    table = [0] * len(nums)
    table[0] = nums[0]                  # only house 0 available
    table[1] = max(nums[0], nums[1])    # two houses — take the richer

    for i in range(2, len(nums)):
        table[i] = max(table[i - 1], table[i - 2] + nums[i])

    return table[-1]

def rob_two_vars(nums: list[int]) -> int:
    """
    Bottom-up with O(1) space: keep only the values you still need.

    Time:  O(n)
    Space: O(1)

    This is the version to be fluent in. Same sliding-window trick as #70's
    two_vars — when you're deciding about house i, everything before i-2 is dead.

    Same classic bug too: update the two variables in the wrong order and you
    clobber one before reading it. Trace [2,1,1,2] by hand before trusting it.
    """
    prev2, prev1 = 0, 0          # the ground, and house -1 (both zero)

    for money in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + money)

    return prev1

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

CASES: list[tuple[str, list[int], int]] = [
    ("normal",        [1, 2, 3, 1],       4),
    ("bigger",        [2, 7, 9, 3, 1],   12),
    ("skip-two trap", [2, 1, 1, 2],       4),   # greedy alternating gives 3 — wrong
    ("far ends",      [5, 1, 1, 5],      10),
    ("all equal",     [4, 4, 4, 4],       8),
    ("two houses",    [2, 3],             3),
    ("single",        [5],                5),
    ("all zeros",     [0, 0, 0],          0),
    ("empty",         [],                 0),
    ("long (n=40)",   list(range(1, 41)), 420),  # kills anything exponential
]

SOLVERS = [
    ("brute_force", rob_brute_force),
    ("memo",        rob_memo),
    ("tabulation",  rob_tabulation),
    ("two_vars",    rob_two_vars),
]

# brute force is O(2^n) — running it on the long case would hang the harness
BRUTE_FORCE_LIMIT = 20

print("198. House Robber")
for label, solver in SOLVERS:
    print(f"\n  {label}")
    for name, nums, expected in CASES:
        if label == "brute_force" and len(nums) > BRUTE_FORCE_LIMIT:
            print(f"    {name:14s}: skipped (too slow — that's the point)")
            continue
        got = solver(list(nums))
        if got is None:
            print(f"    {name:14s}: not implemented")
            continue
        mark = "OK  " if got == expected else "FAIL"
        print(f"    {name:14s}: {mark} got={got} expected={expected}")
