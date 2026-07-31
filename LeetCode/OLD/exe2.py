"""
2. Contains Duplicate — Easy, ~10 min
Given an integer array nums, return True if any value appears at least twice, False if every element is distinct.
Input: nums = [1,2,3,1] → True
Input: nums = [1,2,3,4] → False
"""

def contains_duplicate_brute_force_new(nums: list[int]) -> bool: 
    """
    v1: nested loop
    """

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False

def contains_duplicate_hashmap_new(nums: list[int]) -> bool:
    """
    v2: hashmap
    """
    seen = {}

    for i, num in enumerate(nums):
        if num in seen:
            return True
        seen[num] = i
    return False

def contains_duplicate_set_size_new(nums: list[int]) -> bool:
    """
    v3: Using a set and checking size
    Time: Best/Worst case O(n) — single pass, O(1) average set lookups.
    Space: Best/Worst case O(n) — set holds up to n entries.
    """

    seen = set()
    original_size = len(nums)

    for num in nums:
        seen.add(num)
    if len(seen) != original_size:
        return True
    else: 
        return False
    
def contains_duplicate_sort_new(nums: list[int]) -> bool:
    """
    v5: sorting and working on them
    Time: O(n log n) + O(n) Best case: avg case, worst case O(n log n)
    Space:
    """

    sorted_nums = sorted(nums)

    for i in range(len(sorted_nums) - 1):
        return True if sorted_nums[i] == sorted_nums[i + 1] else False 

    # Timsort: 
    # Time O(n log n)
    # Space new list O(n), but also without, as Timsort internal merge buffer (merge+insertion) leads to O(n) always (a temp buffer)
    nums_sorted = sorted(nums)

    # Time O(n) at worst
    # Space O(n)
    for i in range(len(nums) - 1):
        if nums_sorted[i] == nums_sorted[i + 1]:
            return True
    return False

# ---

def contains_duplicate_brute_force(nums: list[int]) -> bool:
    """
    v1: nested loop, comparing indices, if values are equal, return bool
    Time: O(n^2) 
    Space: O(1)
    """

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
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

def contains_duplicate_set_size_alt(nums: list[int]) -> bool:
    """
    v3: Using a set and checking size
    Time: Best/Worst case O(n) — single pass, O(1) average set lookups.
    Space: Best/Worst case O(n) — set holds up to n entries.
    """
    seen = set(nums)
    return True if len(seen) != len(nums) else False


def contains_duplicate_set(nums: list[int]) -> bool:
    """
    v4: Using a set
    Time: Best/Worst case O(n) — single pass, O(1) average set lookups.
    Space: Worst case O(n) — set holds up to n entries. Best case O(1) — if a duplicate is found early, the set may be small.
    """
    seen = set()

    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

def contains_duplicate_sort(nums: list[int]) -> bool:
    """
    v5: sorting and working on them
    Time: O(n log n) + O(n) Best case: avg case, worst case O(n log n)
    Space:
    """

    # Timsort: 
    # Time O(n log n)
    # Space new list O(n), but also without, as Timsort internal merge buffer (merge+insertion) leads to O(n) always (a temp buffer)
    nums_sorted = sorted(nums)

    # Time O(n) at worst
    # Space O(n)
    for i in range(len(nums) - 1):
        if nums_sorted[i] == nums_sorted[i + 1]:
            return True
    return False


nums = [1,2,3,2]
print(contains_duplicate_brute_force(nums)) 
print(contains_duplicate_hashmap(nums)) 
print(contains_duplicate_set_size(nums)) 
print(contains_duplicate_set(nums)) 
print(contains_duplicate_sort(nums)) 
