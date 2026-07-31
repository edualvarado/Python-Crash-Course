"""
4. Best Time to Buy and Sell Stock — Easy, ~15 min
prices[i] is the stock price on day i. You may buy once and sell once (buy before sell). Return the max profit achievable, or 0 if no profit possible.
Input: prices = [7,1,5,3,6,4]
Output: 5   # buy at 1, sell at 6
"""

def trader_brute_force(prices: list[int]) -> int:
    """
    v1: find minimum through list, find max through list starting in day where minimum can be bought
    Time: O(n^2) - for each day, find the minimum price, then find the maximum price after that day.
    Space: O(1) - no extra space used
    """
    min_price = float('inf')
    max_profit = 0

    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            min_price = prices[i]

            print(f"min_price {min_price}")
            print(f"max_profit {max_profit}")
            print(f"price in j {prices[j]}")

            profit = prices[j] - min_price

            print(f"profit {profit}")

            if profit > max_profit:
                max_profit = profit
                print(f"max_profit changed {max_profit}")

    return max_profit   


def trader_one_pass(prices: list[int]) -> int:
    """
    v2: find minimum through list, find max through list starting in day where minimum can be bought
    Time: O(n) - for each day, find the minimum price, then find the maximum price after that day.
    Space: O(1) - no extra space used
    """
    min_price = float('inf')
    max_profit = 0

    for price in prices:
        if price < min_price:
            min_price = price
            print(f"Min Price updated: {min_price}")
        else:
            print("Price is higher : Buying signal")
            profit = price - min_price
            print(f"profit: {profit}")

            if profit > max_profit:
                max_profit = profit
                print(f"Max Profit updated: {max_profit}")

    return max_profit


def trader_kadane(prices: list[int]) -> int:
    """
    v3: reframe as max subarray sum (Kadane's algorithm) over day-to-day price differences.
    The best profit equals the most profitable contiguous run of daily gains/losses.
    Time: O(n) - single pass over the differences
    Space: O(1) - running sums only, no diffs array materialized
    """
    max_profit = 0
    current = 0

    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        current = max(0, current + diff)
        print(f"diff {diff}, current run {current}")

        if current > max_profit:
            max_profit = current
            print(f"max_profit changed {max_profit}")

    return max_profit


prices = [7,1,5,3,6,4]

"""
[-6, 4, -2, 3, -2]
"""

# prices = [1,10,0,5]

# print(trader_brute_force(prices))
# print(trader_one_pass(prices))
print(trader_kadane(prices))