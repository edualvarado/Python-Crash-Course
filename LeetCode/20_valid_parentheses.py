"""
Valid Parentheses (#20) — Easy

Given a string containing just '(){}[]', determine if the input is valid:
every open bracket is closed by the same type, in the correct order.

Input: "()[]{}" → True
Input: "(]" → False
Input: "([)]" → False
Input: "((()))" → True
"""

def brackets_check_brute_force(brackets: str) -> bool:
    """
    v1: repeatedly remove adjacent matching pairs until none are left, or none can be found
    Time: while best case: no match found, O(1), best case, a match found everytime, the while goes o(n/2) times.
    for loop, if flat "()()...", is the best case, so O(n/2) times. worst case O(n) times if no pattern. If pattern but nested ((())), then quadratic O(n^2) times.
    Space:
    """

    brackets = list(brackets)

    while(brackets):
        found_match = False
        for i in range(len(brackets) - 1):
            if brackets[i] == "(" and brackets[i + 1] == ")" or brackets[i] == "[" and brackets[i + 1] == "]" or brackets[i] == "{" and brackets[i + 1] == "}":
                brackets.pop(i + 1)
                brackets.pop(i)
                found_match = True
                break

        if found_match == False:
            return False
    return True

def brackets_check_stack(brackets: str) -> bool:
    """
    v2. Use hashmap to lookup for value, and a stack (LIFO) queue.
    Time: Worst O(n), Best O(1)
    Space: Worst O(n), Best O(1)
    """
    pairs = {')' : '(', ']' : '[' , '}' : '{'}
    stack = []

    for i, char in enumerate(brackets):
        if char in pairs: # is a closing bracket
            # BUG: stack[-1] is read before `and stack` checks it's non-empty -> IndexError on a leading unmatched closer (e.g. "]")
            if pairs[char] == stack[-1] and stack:
                stack.pop(-1)
            else:
                return False
        else:             # is an opening bracket
            stack.append(char)

    return not stack


test_cases = [
    ("()[]{}", True),   # normal case
    ("(]", False),      # mismatched pair
    ("([)]", False),    # wrong order (interleaved)
    ("((()))", True),   # nested
    ("", True),         # edge case: empty string
    ("(", False),       # edge case: single unmatched open
    ("]", False),       # edge case: single unmatched close
]

print("20. Valid Parentheses")
for s, expected in test_cases:
    got_bf = brackets_check_brute_force(s)
    try:
        got_stack = brackets_check_stack(s)
    except IndexError:
        got_stack = "IndexError"  # bug: stack[-1] is read before checking `stack` is non-empty
    print(f"   {s!r:10s} expected={expected!s:5s}  brute_force={got_bf}  stack={got_stack}")
