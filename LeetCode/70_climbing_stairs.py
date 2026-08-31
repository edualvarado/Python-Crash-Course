"""
70. Climbing Stairs — Easy, ~15 min

You're climbing a staircase with n steps. Each move you can climb 1 or 2 steps.
How many distinct ways can you reach the top?

Input:  n = 2   ->  2      (1+1, 2)
Input:  n = 3   ->  3      (1+1+1, 1+2, 2+1)
Input:  n = 5   ->  8

Narrate before coding:
- Clarify first: what's the answer for n = 0? For n = 1? Is n ever negative?
  LeetCode's constraint is 1 <= n <= 45 — say that out loud, it rules things in
  and out (45 fits in a machine int, and it makes an exponential solution
  obviously untenable).
- Find the recurrence yourself: the LAST move you make is either a 1-step or a
  2-step. So every path to n arrives from exactly one of two places. Which two?
  Write f(n) in terms of smaller f's, and say why the two cases don't overlap.
- Then state the base cases and defend them. f(1) is easy. f(0) is the one an
  interviewer will poke at — argue why it's what you say it is.
- Say what sequence this is. Naming it is worth a point.
- The four functions below are a ladder. Each one removes a specific cost of the
  one above it. Before writing each, say which cost you're removing:
      brute_force  ->  memo         : stop recomputing the same subproblem
      memo         ->  tabulation   : stop paying for the call stack
      tabulation   ->  two_vars     : stop storing values you'll never read again
- For the last one: at the moment you compute f(i), how many earlier values do
  you actually still need? That number is your space.

Run the checklist out loud, starting from step 1:

1. Restate the problem + clarify edge cases
2. State your approach + complexity — before writing code
3. Code
4. Test on a normal case + an edge case
5. State final time/space complexity
"""

"""
n = 5

Last step must be n-1 (4) or n-2 (3), so ...+1 or ...+2

how many ways are there to reach n, in terms of ways to reach n-1 or n-2?

                              f(5)
                 ┌──────────────┴──────────────┐
               f(4)                           f(3)
        ┌───────┴───────┐               ┌───────┴───────┐
      f(3)             f(2)            f(2)             f(1)
   ┌────┴────┐      ┌────┴────┐     ┌────┴────┐
  f(2)      f(1)   f(1)      f(0)  f(1)      f(0)
 ┌──┴──┐
f(1)  f(0)

Naive recursion: O(2ⁿ) time — exactly the exponential blowup the tree shows.
Memoization (top-down, cache each f(k) the first time it's computed): O(n) time, O(n) space for the cache + O(n) recursion stack.

DP (dynamic programming) — a technique for problems that break into overlapping subproblems: instead of recomputing the same subproblem repeatedly (like the exponential tree you just drew), you solve each subproblem once and store the result, then reuse it. That's the "cache repeated calls" idea from before — DP is the general name for that strategy. Two flavors:

- Top-down (memoization): still write it as recursion, but check a cache before recomputing, and store new results into the cache. This is what you already described.
- Bottom-up (tabulation): no recursion at all — start from the base cases and build up in a loop, filling a table until you reach f(n).

On the last question, let me make it concrete. Bottom-up for climbing stairs looks like:

table[0] = 1
table[1] = 1
for i in range(2, n+1):
    table[i] = table[i-1] + table[i-2]
return table[n]

That table array stores every value from f(0) to f(n) — that's O(n) space.

- Memoization (top-down): still recursive, so you pay for the call stack (O(n) stack depth here, plus real risk of stack overflow for large n in languages without tail-call optimization). The cache itself is also O(n) (dict or array). It's usually the easiest thing to write first, since it's just brute-force recursion + a cache check bolted on.
- Tabulation (bottom-up): no recursion, so no call stack cost — you just loop. Because it's iterative, it's also the natural place to spot the O(1) space trick (only keeping prev/prev2) — that optimization is awkward to express in a recursive memoized version, since recursion re-enters the function with the full subproblem history implicitly on the stack.

Summary:


"""


def climb_stairs_brute_force(n: int) -> int:
    """
    Plain recursion, no cache. Recompute every subproblem every time.

    Time:  O(2^n)
    Space: O(n)  -- recursion stack depth

    Write the recurrence directly as code and stop. The point of this one is to
    be wrong in an instructive way: the harness below refuses to run it past
    n = 25, and you should be able to say why that limit exists.
    """
    if n <= 1:
        return 1

    return climb_stairs_brute_force(n - 1) + climb_stairs_brute_force(n - 2)


def climb_stairs_memo(n: int) -> int:
    """
    Top-down DP (memoization): the same recursion, plus a cache.

    Time:  O(n)   -- each subproblem computed once
    Space: O(n)   -- cache, plus O(n) recursion stack

    Two extra lines over brute force: one that checks the cache before doing any
    work, one that stores the result before returning. Decide where the cache
    lives — remember what you learned in #98 about a variable inside a nested
    function being private to a single call.
    """
    cache = {}

    def f(k: int) -> int:
        if k <= 1:
            return 1

        if k in cache:          # already worked this out — just hand it back
            return cache[k]

        cache[k] = f(k - 1) + f(k - 2)    # do the work, and WRITE IT DOWN
        return cache[k]

    return f(n)

def climb_stairs_tabulation(n: int) -> int:
    """
    Bottom-up DP (tabulation): no recursion, fill a table with a loop.

    Time:  O(n)
    Space: O(n)   -- the table

    You already wrote the shape of this in your notes above. Mind the loop
    bounds: check by hand that n = 1 doesn't index off the end of the table.
    """
    table = [1] * (n + 1)        # sets table[0] = 1 and table[1] = 1 at once

    for i in range(2, n + 1):
        table[i] = table[i - 1] + table[i - 2]

    return table[n]

def climb_stairs_two_vars(n: int) -> int:
    """
    Bottom-up with O(1) space: keep only the values you still need.

    Time:  O(n)
    Space: O(1)

    This is the version to be fluent in. The table above is wasteful — when you
    compute table[i], everything below table[i-2] is dead. Replace the table
    with that many variables.

    The classic bug here is updating the two variables in the wrong order and
    clobbering one before you've read it. Trace n = 3 by hand before you trust it.
    """
    prev2, prev1 = 1, 1          # the ground, and stair 1

    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev1 + prev2

    return prev1


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

CASES: list[tuple[str, int, int]] = [
    ("n=1",        1,          1),
    ("n=2",        2,          2),
    ("n=3",        3,          3),
    ("n=4",        4,          5),
    ("n=5",        5,          8),
    ("n=10",      10,         89),
    ("n=0",        0,          1),   # edge: clarify this one with the interviewer
    ("n=25",      25,     242785),
    ("n=45",      45, 1836311903),   # LeetCode's max — kills anything exponential
]

SOLVERS = [
    ("brute_force", climb_stairs_brute_force),
    ("memo",        climb_stairs_memo),
    ("tabulation",  climb_stairs_tabulation),
    ("two_vars",    climb_stairs_two_vars),
]

# brute force is O(2^n) — running it on the big cases would hang the harness
BRUTE_FORCE_LIMIT = 25

print("70. Climbing Stairs")
for label, solver in SOLVERS:
    print(f"\n  {label}")
    for name, n, expected in CASES:
        if label == "brute_force" and n > BRUTE_FORCE_LIMIT:
            print(f"    {name:6s}: skipped (too slow — that's the point)")
            continue
        got = solver(n)
        if got is None:
            print(f"    {name:6s}: not implemented")
            continue
        mark = "OK  " if got == expected else "FAIL"
        print(f"    {name:6s}: {mark} got={got} expected={expected}")
