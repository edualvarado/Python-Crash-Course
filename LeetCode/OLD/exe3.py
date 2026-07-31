"""
3. Valid Parentheses — Easy, ~15 min
Given a string containing just '(){}[]', determine if the input is valid: every open bracket is closed by the same type, in the correct order.
Input: "()[]{}" → True
Input: "(]" → False
Input: "([)]" → False
Input: "((()))" → True

"""


def brackets_check_brute_force_new(brackets: str) -> bool:
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

def brackets_check_stack_new(brackets: str) -> bool:
    """
    v2. Use hashmap to lookup for value, and a stack (LIFO) queue.
    Time: Worst O(n), Best O(1)
    Space: Worst O(n), Best O(1)
    """
    pairs = {')' : '(', ']' : '[' , '}' : '{'}
    stack = []

    for i, char in enumerate(brackets):
        if char in pairs: # is a closing bracket
            if pairs[char] == stack[-1] and stack:
                stack.pop(-1)
            else:
                return False
        else:             # is an opening bracket
            stack.append(char)             

    return not stack

# ---

def brackets_check_brute_force(brackets: str) -> bool:
    """
    v1: repeatedly remove adjacent matching pairs until none are left, or none can be found
    Time: while best case: no match found, O(1), best case, a match found everytime, the while goes o(n/2) times.
    for loop, if flat "()()...", is the best case, so O(n/2) times. worst case O(n) times if no pattern. If pattern but nested ((())), then quadratic O(n^2) times.
    Space: 
    """

    brackets = list(brackets)
    print(brackets)

    while brackets:
        found_match = False
        for i in range(len(brackets) - 1):
            print(f"found? {found_match}")
            if brackets[i] == "(" and brackets[i + 1] == ")" or brackets[i] == "[" and brackets[i + 1] == "]" or brackets[i] == "{" and brackets[i + 1] == "}":
                brackets.pop(i + 1)
                brackets.pop(i)
                found_match = True
                print(f"found? {found_match}")
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

    if len(brackets) % 2 != 0:
        return False

    for char in brackets:
        if char in pairs: # is a closing bracket
            print(f"closing {char}")
            if stack and stack[-1] == pairs[char]:
                print(f"previous {stack[-1]} corresponds to the opening in dict {pairs[char]}")
                print(f"removing {stack[-1]}")
                stack.pop() 
                print(f"stack {stack}")
            else:
                print(f"adding {char} - mismatch")
                return False
        else: # is an opening bracket
            print(f"adding opening {char}")
            stack.append(char)
            print(f"stack {stack}")

    # if stack == []:
    #     return True
    # else:
    #     return False

    return not stack


brackets = '()[]{}'
# brackets = '(])'
# brackets = ')('
# brackets = ']'
# brackets = '((('
# brackets = ']]]'


print(brackets_check_brute_force_new(brackets))
# print(brackets_check_stack(brackets))