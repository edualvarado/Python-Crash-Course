"""
Contains Duplicate (#217) — Easy

Given an integer array nums, return True if any value appears at least twice,
and False if every element is distinct.

Input: nums = [1,2,3,1] → True
Input: nums = [1,2,3,4] → False
"""

def contains_duplicate_brute_force(nums: list[int]) -> bool:
    """
    Brute force: compare every pair of elements.

    Time:  O(n^2)
    Space: O(1)
    """

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False

def contains_duplicate_set(nums: list[int]) -> bool:
    """
    Optimized: track seen values in a set.
    Membership check (`in`) is O(1) average on a set vs O(n) on a list —
    that's the whole reason this beats the brute force.

    Time:  O(n)
    Space: O(n)
    """

    seen = set(nums)

    return True if len(seen) != len(nums) else False

def contains_duplicate_sort(nums: list[int]) -> bool:
    """
    Alternative: sort, then check adjacent elements for equality.
    Trades time for space vs the set version — no extra data structure
    (aside from what the sort itself costs), but O(n log n) instead of O(n).

    Time:  O(n log n)
    Space: O(1) extra, ignoring sort's own space
    """

    sorted_nums = sorted(nums)

    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i] == sorted_nums[i + 1]:
            return True
    return False



def contains_duplicate_hashmap(nums: list[int]) -> bool:
    """
    v2: hashmap
    Time: O(n)
    Space: O(n)
    """
    seen = {}

    for i, num in enumerate(nums):
        if num in seen:
            # Duplicated found
            return True
        seen[num] = i
    return False

def contains_duplicate_set_size(nums: list[int]) -> bool:
    """
    v3: Using a set and checking size
    Time: Best/Worst case O(n) — single pass, O(1) average set lookups.
    Space: Best/Worst case O(n) — set holds up to n entries.
    """
    seen = set()
    original_size = len(nums)

    for num in nums:
        seen.add(num)
    return True if len(seen) != original_size else False

def contains_duplicate_set_early_exit(nums: list[int]) -> bool:
    """
    v4: Using a set, early exit
    Time: Best/Worst case O(n) — single pass, O(1) average set lookups.
    Space: Worst case O(n) — set holds up to n entries. Best case O(1) — if a duplicate is found early, the set may be small.
    """
    seen = set()

    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False


nums1 = [1, 2, 3, 1]   # normal case -> True
nums2 = [1, 2, 3, 4]   # normal case -> False
nums3: list[int] = []  # edge case: empty -> False
nums4 = [7]            # edge case: single element -> False
nums5 = [5, 5, 5, 5]   # edge case: all duplicates -> True

print("217. Contains Duplicate")
for name, nums in [("duplicate", nums1), ("distinct", nums2), ("empty", nums3),
                    ("single", nums4), ("all same", nums5)]:
    print(f"   {name:9s}: brute force={contains_duplicate_brute_force(nums)}  "
          f"set={contains_duplicate_set(nums)}  sort={contains_duplicate_sort(nums)}  "
          f"hashmap={contains_duplicate_hashmap(nums)}  "
          f"set_size={contains_duplicate_set_size(nums)}  "
          f"set_early_exit={contains_duplicate_set_early_exit(nums)}")
