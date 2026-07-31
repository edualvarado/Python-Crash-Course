"""
1. Two Sum — Easy, ~15 min
Given an array of integers nums and an integer target, return the indices of the two numbers that add up to target. Assume exactly one solution, can't use the same element twice.
Input: nums = [2,7,11,15], target = 9
Output: [0,1]   # nums[0] + nums[1] == 9
"""

def two_sum_brute_force_new(nums: list[int], target: int) -> list[int]:
    """
    v1: nested loop
    Time: O(n^2)
    Space: O(1)
    """

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return None

def two_sum_hashmap_new(nums: list[int], target: int) -> list[int]:
    """
    v2: hashmap
    Time:
    Space:
    """
    seen = {}

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return None

# ---

def two_sum_brute_force(nums: list[int], target: int) -> list[int]:
    """
    v1: checking each pair of indices
    Time: O(n^2) - nested over all pairs, quadratic
    Space: O(1) - no extra data structure
    """

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def two_sum_hashmap(nums: list[int], target: int) -> list[int]:
    """
    v2: using a hashmap, look at the complement as we go
    Time: O(n): each lookup is O(1), loop makes it O(n)
    Space: O(n) - seen grows with input size
    """
    seen={}
    for i, num in enumerate(nums):
        complement = target - num 
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


nums = [2, 7, 11, 15]
target = 9

print("1. Two Sum")
print("   brute force:", two_sum_brute_force_new(nums, target))
print("   hash map:   ", two_sum_hashmap_new(nums, target))