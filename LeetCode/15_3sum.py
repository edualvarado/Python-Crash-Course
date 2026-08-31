"""
3Sum (#15) — Medium

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]]
such that i != j, i != k, j != k, and nums[i] + nums[j] + nums[k] == 0.

The solution set must NOT contain duplicate triplets. Order of the triplets, and
order within a triplet, does not matter.

Input:  nums = [-1,0,1,2,-1,-4]  ->  [[-1,-1,2],[-1,0,1]]
Input:  nums = [0,1,1]           ->  []
Input:  nums = [0,0,0]           ->  [[0,0,0]]

Narrate before coding:
- Two distinct sub-problems here: (1) finding triplets that sum to 0,
  (2) not emitting the same triplet twice. Decide how you handle (2) BEFORE
  you start typing — it's what this problem actually tests.
- What does 3Sum reduce to once you fix the first element?
- Constraint worth stating out loud: n can be ~3000, so O(n^3) is out.
"""


def three_sum_brute_force(nums: list[int]) -> list[list[int]]:
    """
    Brute force: check every triplet of indices.

    Time:  O(n^3)
    Space: O(1) extra, ignoring the output

    Say out loud why this is too slow at n = 3000 before moving on.
    """
    # TODO
    triplet = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            for k in range(j + 1, len(nums)):
                if nums[i] + nums[j] + nums[k] == 0:
                    key = tuple(sorted([nums[i], nums[j], nums[k]]))
                    triplet.add(key)

    return [list(t) for t in triplet]


def three_sum_hashset(nums: list[int]) -> list[list[int]]:
    """
    Hash-based: reduce to a 2Sum lookup for each fixed first element.

    Time:  O(n^2)
    Space: O(n)

    Target complexity to hit — how you get there is yours to work out.
    """
    # TODO
    nums = sorted(nums)
    triplets = []

    for i in range(len(nums)):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        target = -nums[i]
        seen = set()
        j = i + 1

        while j < len(nums):
            complement = target - nums[j]
            if complement in seen:
                triplets.append([nums[i], complement, nums[j]])
                while j + 1 < len(nums) and nums[j] == nums[j + 1]:   # dedup the pair
                    j += 1
            seen.add(nums[j])
            j += 1
    
    return triplets

def three_sum_two_pointers(nums: list[int]) -> list[list[int]]:
    """
    Sort + two pointers (the Day 2 pattern, applied one level deeper).

    Time:  O(n^2)  -- O(n log n) sort + O(n^2) scan
    Space: O(1) extra, ignoring the output and the sort

    This is the version to be fluent in — it's what an interviewer expects
    and the duplicate handling falls out of the sorted order for free.

    [-1,0,1,2,-1,-4]
    [-4, -1, -1, 0, 1, 2]

    1) left = -4, right = 2, target = 2
        

    """
    # TODO
    nums = sorted(nums)
    n = len(nums)
    triplets = []

    for i in range(n - 2):
        if nums[i] > 0:                       # sorted: three positives can't hit 0
            break
        if i > 0 and nums[i] == nums[i - 1]:  # don't fix the same value twice
            continue

        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1                     # need a bigger sum
            elif total > 0:
                right -= 1                    # need a smaller sum
            else:
                triplets.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1                 # skip repeats of the value just used

    return triplets




# ---------------------------------------------------------------------------
# Test harness — comparison is order-insensitive (triplets and their contents)
# ---------------------------------------------------------------------------

def _normalize(triplets: list[list[int]] | None) -> set[tuple[int, ...]] | None:
    """Canonical form so [[-1,0,1]] and [[1,-1,0]] compare equal."""
    if triplets is None:
        return None
    return {tuple(sorted(t)) for t in triplets}


CASES: list[tuple[str, list[int], list[list[int]]]] = [
    ("normal",          [-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
    ("dup-heavy",       [-2, 0, 1, 1, 2],      [[-2, 0, 2], [-2, 1, 1]]),
    ("no triplet",      [0, 1, 1],             []),
    ("all zeros",       [0, 0, 0],             [[0, 0, 0]]),
    ("all zeros x4",    [0, 0, 0, 0],          [[0, 0, 0]]),   # must NOT emit it 4x
    ("all positive",    [1, 2, 3, 4],          []),
    ("empty",           [],                    []),
    ("too short",       [0, 0],                []),
    ("single",          [0],                   []),
]

SOLVERS = [
    ("brute_force",  three_sum_brute_force),
    ("hashset",      three_sum_hashset),
    ("two_pointers", three_sum_two_pointers),
]

print("15. 3Sum")
for label, solver in SOLVERS:
    print(f"\n  {label}")
    for name, nums, expected in CASES:
        raw = solver(list(nums))
        if raw is None:
            print(f"    {name:14s}: not implemented")
            continue
        got, want = _normalize(raw), _normalize(expected)
        if got != want:
            mark = "FAIL"
        elif len(raw) != len(got):
            # right triplets, but emitted more than once — the actual trap of #15
            mark = f"DUP ({len(raw)} returned, {len(got)} distinct)"
        else:
            mark = "OK  "
        print(f"    {name:14s}: {mark} got={sorted(got)} expected={sorted(want)}")
