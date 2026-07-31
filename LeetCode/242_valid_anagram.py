"""
▎ Given two strings s and t, return true if t is an anagram of s, and false otherwise.
▎ An anagram is a word formed by rearranging the letters of another, using all original letters exactly once.

Example: s = "anagram", t = "nagaram" → true. s = "rat", t = "car" → false.

"""

# NOT WORKING
def is_anagram_v1(s: str, t: str) -> bool:
    """
    v1: We just check each char in the other word and compare lengths.
    Time: O(n)
    Space: O(1)
    """

    if len(s) != len(t):
        return False

    for char in s:
        if char not in t:
            return False

    return True

# 2. WORKING
def is_anagram_v2(s: str, t: str) -> bool:
    """
    v3: Use of dictionaries to count character frequencies
    Time: O(n)
    Space: O(n)
    """

    total = 0

    # lower scans all chars -> O(n), replace scans all chars -> O(n). Time/Space are O(n)
    s = s.lower().replace(" ", "")
    t = t.lower().replace(" ", "")

    # len(s) != len(t) -> O(1)
    if len(s) != len(t):
        return False

    # O(1) for now
    dict_s = {}

    # for char in s:
    #     if char not in dict_s:
    #         dict_s[char] = 1
    #     else:
    #         dict_s[char] += 1

    # Loop over all chars in s: O(n)
    for char in s:
        # get() and assignment is O(1), so this is O(n) overall for time
        # Space is O(k), where k < n. If all letters are unique, then k = n. If all letters are the same, then k = 1. So space is O(n) in worst case.
        dict_s[char] = dict_s.get(char, 0) + 1

    print(dict_s)

    # O(n) for loop over all chars in t
    for char in t:   
        # O(1) for lookup, so O(n) overall for time. Space is O(1) since we are not creating any new data structures.
        if char not in dict_s:
            return False
        # O(1) for decrement and deletion, so O(n) overall for time. Space is O(1) since we are not creating any new data structures.
        dict_s[char] -= 1
        if dict_s[char] == 0:
            del dict_s[char]

    print(dict_s)

    return len(dict_s) == 0
      
# BEST: WORKING
def is_anagram_v3(s: str, t: str) -> bool:
    """
    v4: Alphabet
    Time: O(n)
    Space: O(1)
    """

    # Don't do, remove O(n) 
    s = s.lower().replace(" ", "")
    t = t.lower().replace(" ", "")

    if len(s) != len(t):
        return False

    counts = [0] * 26

    # We can build an array index 0-25 if we take the unicode of the char and substract ord("a")

    for i in range(len(s)):
        idx_s = ord(s[i]) - ord('a')
        counts[idx_s] += 1
        idx_t = ord(t[i]) - ord('a')
        counts[idx_t] -= 1

    # Boring way
    for c in counts:
        if c != 0:
            return False

    # Nice way
    return all(c == 0 for c in counts)


# 1. WORKING
def is_anagram_v4(s: str, t: str) -> bool:
    """
    v5: Use of sorting
    Time: Timsort is O(n log n)
    Space: O(n)
    """

    s = s.lower().replace(" ", "")
    t = t.lower().replace(" ", "")

    if len(s) != len(t):
        return False
    
    return sorted(s) == sorted(t)

# final API: WORKING
from collections import Counter
def is_anagram_v5(s: str, t: str) -> bool:
    """
    v6: Use of Counter from collections
    Time: O(n)
    Space: O(n)
    """

    return Counter(s) == Counter(t)


# 1. WORKING
def is_anagram_v4_naive(s: str, t: str) -> bool:
    """
    v5: Use of sorting
    Time: Timsort is O(n log n)
    Space: O(n)
    """

    s = s.lower().replace(" ", "")
    t = t.lower().replace(" ", "")

    if len(s) != len(t):
        return False 

    return sorted(s) == sorted(t)

    # s = sorted(s)
    # t = sorted(t)

    # for i in range(len(s)):
    #     if s[i] != t[i]:
    #         return False
    # return True


def is_anagram_v2_naive(s: str, t: str) -> bool:
    """
    v3: Use of dictionaries to count character frequencies
    Time: O(n)
    Space: O(n)
    """

    total = 0

    # lower scans all chars -> O(n), replace scans all chars -> O(n). Time/Space are O(n)
    s = s.lower().replace(" ", "")
    t = t.lower().replace(" ", "")

    # len(s) != len(t) -> O(1)
    if len(s) != len(t):
        return False

    # O(1) for now
    dict_s = {}

    # for char in s:
    #     if char not in dict_s:
    #         dict_s[char] = 1
    #     else:
    #         dict_s[char] += 1

    # Loop over all chars in s: O(n)
    for char in s:
        # get() and assignment is O(1), so this is O(n) overall for time
        # Space is O(k), where k < n. If all letters are unique, then k = n. If all letters are the same, then k = 1. So space is O(n) in worst case.
        dict_s[char] = dict_s.get(char, 0) + 1

    print(dict_s)

    # O(n) for loop over all chars in t
    for char in t:
        # get() and assignment is O(1), so this is O(n) overall for time
        # Space is O(k), where k < n. If all letters are unique, then k = n. If all letters are the same, then k = 1. So space is O(n) in worst case.
        dict_s[char] = dict_s.get(char, 0) - 1

    print(dict_s)

    return all(value == 0 for value in dict_s.values())

# Test cases
print(is_anagram_v2_naive("anagram", "nagaram"))  # True
print(is_anagram_v2_naive("rat", "car"))          # False
print(is_anagram_v2_naive("aab", "abb"))          # False
print(is_anagram_v2_naive("Aab", "aab"))          # True
print(is_anagram_v2_naive("listen ole", "silent leo"))    # True

