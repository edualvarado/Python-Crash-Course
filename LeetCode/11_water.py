"""
Given n non-negative integers height[0], height[1], ..., height[n-1], where each represents a point at coordinate (i, height[i]), find two lines that together with the x-axis form a container that holds the most water. Return the maximum amount of water it can contain. (Note: you can't slant the container — it's vertical lines only.)

Example: height = [1,8,6,2,5,4,8,3,7] → 49 (lines at index 1, height 8, and index 8, height 7 → width 7 × min(8,7)=7 → 49).

This is the converging two-pointers pattern from earlier — good one to apply what we just covered.
"""

"""
Volume would be (j-i) * min(height[i], height[j]) where i and j are the two indices of the lines. We want to maximize this volume.
"""


def max_area_brute_force(height: list[int]) -> int:
    """
    v1: brute force, check all pairs of lines (i, j) and calculate the volume. Keep track of the maximum volume found.
    Time: O(n^2) - for each line, we check all other lines to calculate the volume.
    Space: O(1) - no extra space used.  
    """

    max_vol = 0

    for i in range(len(height)):
        for j in range(i + 1, len(height)):            
            bin_width = j - i
            bin_height = min(height[i], height[j])
            bin_vol = bin_width * bin_height

            if bin_vol > max_vol:
                max_vol = bin_vol

    return max_vol

# def max_area_sliding_window(height: list[int]) -> int:
#     """
#     Sliding window maintains an aggregate over a contiguous range (running sum, char counts) that gets incrementally updated, and you grow/shrink the window because some validity condition over that range was violated.
#     """

#     i = 0

#     for j in range(len(height)):

#         bin_width = j - i
#         bin_height = min(height[i], height[j])
#         bin_vol = bin_width * bin_height

def max_area_pointers(height: list[int]) -> int:
    """
    v2: using two pointers, look to the cap height, move in that direction
    Time: O(n)
    Space: O(1)
    """

    left = 0
    right = len(height) - 1
    max_vol = 0

    while left < right: 
        vol = (right - left) * min(height[left], height[right])
        if vol > max_vol:
            max_vol = vol

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_vol

height = [1,8,6,2,5,4,8,3,7]
print(max_area_pointers(height))