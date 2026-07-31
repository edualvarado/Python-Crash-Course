# Two pointers & sliding window

Two pointers — two indices over a sequence, moving independently. Two flavors:
- Converging (opposite ends → meet in middle): sorted array, need a pair/triplet hitting a target (sum, max area, palindrome check). Reach for it when the array is sorted or the answer only cares about combining a low and high element.
- Fast/slow: cycle detection, finding middle of linked list.

```python
left, right = 0, len(arr) - 1
while left < right:
    if some_condition(arr[left], arr[right]):
        # record/update answer
        left += 1
    else:
        right -= 1
```

Sliding window — a special case: both pointers only move rightward, defining a contiguous range. Reach for it when the problem is about a contiguous substring/subarray and you can incrementally update the window's state as you expand or shrink it, instead of recomputing from scratch.

```python
window = {}  # or set, running sum, etc.
left = 0
best = 0
for right in range(len(s)):
    # expand: add s[right] to window
    while window_invalid():        # shrink until valid again
        # remove s[left] from window
        left += 1
    best = max(best, right - left + 1)
```

One-liner for the interviewer: "Two pointers: I converge from both ends when I can discard one side based on a comparison; sliding window is two pointers that only move forward, maintaining a contiguous range with incrementally-updated state."

Maps directly onto today's two problems: #11 (Container With Most Water) is converging two pointers, #3 (Longest Substring Without Repeating Characters) is sliding window.
