"""
Problem: You're climbing a staircase with n steps. Each move you can climb 1 or 2 steps. How many distinct ways can you reach the top?

Example: n = 3 → answer 3 (1+1+1, 1+2, 2+1).
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

