"""
5. Longest Substring Without Repeating Characters — Medium, ~25 min
Given a string s, find the length of the longest substring without repeating characters.
Input: "abcabcbb" → 3   ("abc")
Input: "bbbbb" → 1
Input: "pwwkew" → 3   ("wke")
"""

def length_of_longest_substring(s: str) -> int:
    """
    v1: convert to list,
    """
    s = list(s)
    seen = set()
    counter = 1
    max_counter = 1

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

    Time: O(n) - right moves n times total; left moves at most n times total
          across the whole run, so total work is O(n), not O(n^2).
    Space: O(min(n, charset size)) - seen holds at most one window's worth
           of unique characters.
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

# def length_of_longest_substring_window_self(s: str) -> int:
#     """
#     Same as before but implementing it myself
#     """

#     seen = set()
#     left = 0

#     max_len = 0

#     for right in range(len(s)):
#         while s[right] in seen:
#             seen.remove(s[left])
#             left += 1
#         else:
#             seen.add(s[right])
#             max_len += 1

#     return max_len



input = "abcdabcde"
input = "abcdcbcde"

input = "bbbbb"
# input = "pwwkew"
# input = "a"
# input = "abcdabcde"
# input = "dvdf"

print(length_of_longest_substring(input))
print(length_of_longest_substring_window(input))
