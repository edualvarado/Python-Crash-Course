"""
▎ Given a string s, find the length of the longest substring without repeating characters.

Example: "abcabcbb" → 3 ("abc"). "bbbbb" → 1. "pwwkew" → 3 ("wke").

Run the checklist out loud, starting from step 1:

1. Restate the problem + clarify edge cases (empty string? all same char? unicode/ASCII assumption?)
2. State your approach + complexity — before writing code
3. Code
4. Test on a normal case + an edge case
5. State final time/space complexity
"""

def length_of_longest_substring(s: str) -> int:
    """
    v1: convert to list,
    """
    s = list(s)
    seen = set()
    counter = 1
    max_counter = 1  # BUG: for an empty string the loop below never runs, so this stays 1 instead of 0

    print(s)

    for i in range(len(s) - 1):
        print("Current is " + s[i])
        seen.add(s[i])
        print(f"Adding to set: Current set is {seen}")
        print("Looking into " + s[i + 1] + " is in set?")
        if s[i + 1] not in seen:
            counter += 1
            print(f"No! Counter++ is: {counter}")
        else:
            counter = 1
            seen.clear()
            print(f"Yes! Reset counter to: {counter}")
        max_counter = max(max_counter, counter)

    return max_counter


def length_of_longest_substring_window(s: str) -> int:
    """
    v2: sliding window with two pointers (left, right). `seen` only ever holds characters inside the current window [left, right]. 
    When s[right] is already in the window, shrink from the left - removing characters from `seen` one at a time - until the duplicate is gone, then add s[right].

    Time: O(n) - right moves n times total; left moves at most n times total across the whole run, so total work is O(n), not O(n^2). Total work = (n increments of right) + (at most n increments of left, summed across all inner while-loops combined) = O(n) + O(n) = O(n).
    Space: O(min(n, charset size)) - seen holds at most one window's worth of unique characters.
    """
    seen = set()
    left = 0
    max_len = 0

    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left])
            left += 1
        seen.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len


test_cases = [
    ("abcabcbb", 3),
    ("bbbbb", 1),
    ("pwwkew", 3),
    ("", 0),   # edge case: empty string
    ("a", 1),  # edge case: single character
]

print("3. Longest Substring Without Repeating Characters")
for s, expected in test_cases:
    got_v1 = length_of_longest_substring(s)
    got_window = length_of_longest_substring_window(s)
    print(f"   {s!r:12s} expected={expected}  v1={got_v1}  window={got_window}")
