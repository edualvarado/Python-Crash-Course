"""
Two Sum (#1)

Given an array of integers nums and an integer target, return indices of the two numbers that add up to target. Assume exactly one solution, can't use the same element twice.
"""

def two_sum(nums: list[int], target: int) -> list[int]:
    """
    V1: Nested loop for, check if sum equals target, return indices if success.
    Time: O(n^2)
    Space: O(1)
    """

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def two_sum_hashmap(nums: list[int], target: int) -> list[int]:
    """
    V2: Hashmap
    Time: O(n)
    Space: O(n)

    input: [2,5,6,8] t=7

    1) for 0, 2: complement = 5, seen={} -> seen{2: 0}
    2) for 1, 5: complement = 2, seen{2: 0} 

    """

    seen = {}

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

nums1, target1 = [2, 7, 11, 15], 9   # normal case -> [0, 1]
nums2, target2 = [1, 1, 1, 1], 2     # edge case: duplicates -> [0, 1]

print("1. Two Sum")
print("   normal case")
print("     brute force:", two_sum(nums1, target1))
print("     hash map:   ", two_sum_hashmap(nums1, target1))
print("   duplicates edge case")
print("     brute force:", two_sum(nums2, target2))
print("     hash map:   ", two_sum_hashmap(nums2, target2))