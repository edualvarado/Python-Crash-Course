"""
Remove Element (#27) — Easy

Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.

Consider the number of elements in nums which are not equal to val be k, to get accepted, you need to do the following things:

Change the array nums such that the first k elements of nums contain the elements which are not equal to val. The remaining elements of nums are not important as well as the size of nums.
Return k.

Custom Judge:
The judge will test your solution with the following code:

int[] nums = [...]; // Input array
int val = ...; // Value to remove
int[] expectedNums = [...]; // The expected answer with correct length.
                            // It is sorted with no values equaling val.

int k = removeElement(nums, val); // Calls your implementation

assert k == expectedNums.length;
sort(nums, 0, k); // Sort the first k elements of nums
for (int i = 0; i < actualLength; i++) {
    assert nums[i] == expectedNums[i];
}
If all assertions pass, then your solution will be accepted.

Example 1:
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 2.
It does not matter what you leave beyond the returned k (hence they are underscores).

Example 2:
Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums containing 0, 0, 1, 3, and 4.
Note that the five elements can be returned in any order.
It does not matter what you leave beyond the returned k (hence they are underscores).
"""

def removeElement(nums: list[int], val: int) -> int:
    """
    v1: brute force - scan the "logical" array (length k). Whenever nums[i] equals
    val, shift every element after it one slot to the left (overwriting it) and
    shrink k by one, without advancing i, since a new value just slid into position i
    and needs to be checked too. If nums[i] doesn't match, move on.
    Time: O(n^2) worst case - up to n removals, each shifting up to n elements.
    Space: O(1) - shifting happens within nums itself.
    """
    k = len(nums)
    i = 0

    while i < k:
        if nums[i] == val:
            for j in range(i, k - 1):
                nums[j] = nums[j + 1]
            k -= 1
        else:
            i += 1

    return k


def removeElement_two_pointer(nums: list[int], val: int) -> int:
    """
    v2: two pointers - i scans every element once, k tracks where the next
    "keep" value should be written. Whenever nums[i] != val, copy it down to
    nums[k] and advance k. Matches are simply skipped over, never shifted.
    Time: O(n) - single pass, each element copied at most once.
    Space: O(1) - overwrites nums in place.
    """
    k = 0

    for i in range(len(nums)):
        if nums[i] != val:
            nums[k] = nums[i]
            k += 1

    return k


def removeElement_list_remove(nums: list[int], val: int) -> int:
    """
    v3: naive removal via list.remove() - repeatedly find and remove the first
    occurrence of val until none remain, then return the new length. Reads as
    "just pop it", but list.remove() re-scans nums from the start to find val
    and then shifts every element after it left by one - both costs the other
    two versions make explicit are still happening here, just hidden behind a
    single line.
    Time: O(n^2) worst case - up to n removals, each an O(n) scan + O(n) shift.
    Space: O(1) extra - list.remove() mutates nums in place.
    """
    while val in nums:
        nums.remove(val)

    return len(nums)


"""
Confirmed still correct — this is the exact same removeElement I already tested (all 6 cases passed), nothing's changed. Let me trace it properly this time, matching what the code actually does: the shift happens when nums[i] == val, not the other way around like I wrongly described before.

nums = [3, 2, 2, 3], val = 3. Start: k = len(nums) = 4, i = 0.

┌──────┬─────┬─────┬─────────┬────────────┬──────────────────────────────────────┬──────────┬───────┬───────┐
│ step │  i  │  k  │ nums[i] │ nums[i] == │                action                │  array   │  k    │  i    │
│      │     │     │         │    val?    │                                      │  after   │ after │ after │
├──────┼─────┼─────┼─────────┼────────────┼──────────────────────────────────────┼──────────┼───────┼───────┤
│ 1    │ 0   │ 4   │ 3       │ yes        │ shift everything after index 0 left  │ [2, 2,   │ 3     │ stays │
│      │     │     │         │            │ by one, k -= 1                       │ 3, 3]    │       │  0    │
├──────┼─────┼─────┼─────────┼────────────┼──────────────────────────────────────┼──────────┼───────┼───────┤
│ 2    │ 0   │ 3   │ 2       │ no         │ just move on, i += 1                 │ [2, 2,   │ 3     │ 1     │
│      │     │     │         │            │                                      │ 3, 3]    │       │       │
├──────┼─────┼─────┼─────────┼────────────┼──────────────────────────────────────┼──────────┼───────┼───────┤
│ 3    │ 1   │ 3   │ 2       │ no         │ move on, i += 1                      │ [2, 2,   │ 3     │ 2     │
│      │     │     │         │            │                                      │ 3, 3]    │       │       │
├──────┼─────┼─────┼─────────┼────────────┼──────────────────────────────────────┼──────────┼───────┼───────┤
│ 4    │ 2   │ 3   │ 3       │ yes        │ shift (nothing left to shift — it's  │ [2, 2,   │ 2     │ stays │
│      │     │     │         │            │ the last logical slot), k -= 1       │ 3, 3]    │       │  2    │
├──────┼─────┼─────┼─────────┼────────────┼──────────────────────────────────────┼──────────┼───────┼───────┤
│ —    │ 2   │ 2   │ —       │ —          │ i < k is 2 < 2 → false, loop ends    │          │       │       │
└──────┴─────┴─────┴─────────┴────────────┴──────────────────────────────────────┴──────────┴───────┴───────┘

Return k = 2. First k elements: nums[0:2] = [2, 2]. Correct.

Key thing to notice, since it's the opposite of what I said earlier: the match (nums[i] == val) is the trigger to act — that's the whole point of this version, since val is exactly what needs to disappear, and "disappear" here means "get physically overwritten by shifting everything after it left." Notice i only stays put right after a shift (step 1→2, step 4→end) — because a new value just slid into position i and hasn't been checked yet. i only advances when the current element is confirmed safe to leave alone (steps 2 and 3).
"""

test_cases = [
    ([3, 2, 2, 3], 3, 2, [2, 2]),                  # normal case (example 1)
    ([0, 1, 2, 2, 3, 0, 4, 2], 2, 5, None),         # example 2 (order-independent, checked by set below)
    ([], 5, 0, []),                                 # edge case: empty array
    ([1, 1, 1], 2, 3, [1, 1, 1]),                   # edge case: val not present
]

def check(k, prefix, val, expected_k, expected_prefix):
    if expected_prefix is None:
        return k == expected_k and val not in prefix
    return k == expected_k and sorted(prefix) == sorted(expected_prefix)


print("27. Remove Element")
for nums, val, expected_k, expected_prefix in test_cases:
    nums_bf = list(nums)
    k_bf = removeElement(nums_bf, val)
    prefix_bf = nums_bf[:k_bf]
    ok_bf = check(k_bf, prefix_bf, val, expected_k, expected_prefix)

    nums_tp = list(nums)
    k_tp = removeElement_two_pointer(nums_tp, val)
    prefix_tp = nums_tp[:k_tp]
    ok_tp = check(k_tp, prefix_tp, val, expected_k, expected_prefix)

    nums_rm = list(nums)
    k_rm = removeElement_list_remove(nums_rm, val)
    prefix_rm = nums_rm[:k_rm]
    ok_rm = check(k_rm, prefix_rm, val, expected_k, expected_prefix)

    print(f"   val={val}  (expected k={expected_k})")
    print(f"     brute force: k={k_bf}  prefix={prefix_bf}  {'OK' if ok_bf else 'MISMATCH'}")
    print(f"     two pointer: k={k_tp}  prefix={prefix_tp}  {'OK' if ok_tp else 'MISMATCH'}")
    print(f"     list.remove: k={k_rm}  prefix={prefix_rm}  {'OK' if ok_rm else 'MISMATCH'}")
