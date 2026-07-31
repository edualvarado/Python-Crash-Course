"""
You are writing the Google Search unit converter. You are given a list of unit conversion factors and a query. The conversion factor is the value needed to convert from unit A to unit B.
Write a function that takes a list of conversion factors and 2 units (a and b) and outputs the conversion factor from a -> b.

[[“meter”, “centmenter”, 100],[“inch”, “centmenter”, 2.56],[“kilometer”, “inch”,4333232]] , “meter”, “kilometer”
"""

"""
Examples:
Input: [["m", "km", 1000], ["m", "cm", 100]], "km", "cm"

unit1 = unit2 * factor
m = km * 1000
m = cm * 100

km = m * 1/1000 = m * 0.001
cm = m * 1/100 = m * 0.01

weight(A→B) x weight(B→A) = 1

From cm to km: 
-- from cm to m (m = cm * 100) and from m to km (km = m * 1/1000 = m * 0.001)
Result: km = cm * 100 * 0.001

[  ] -1000->  [ ] -0.01-> [  ]
 km            m           cm
[  ] <-0.001- [ ] <-100-  [  ]
"""
"""
Walk through your own example:
conversion_factors = [["km", "m", 0.001], ["m", "cm", 0.01]]
Tuple-dict version:
graph = {("km","m"): 0.001, ("m","km"): 1000, ("m","cm"): 0.01, ("cm","m"): 100}
Now say the query is (km, cm) — no direct edge exists, DFS has to search. It starts at km and needs: "give me the list of km's neighbors." With this structure, there's no such lookup — you'd have to scan every key in the whole dict checking whether the first element equals "km":
neighbors = [(b, w) for (a, b), w in graph.items() if a == "km"]
And you'd have to do that scan again at every single node DFS visits, not just once. That turns each "get neighbors" step from O(1)-ish into O(total edges), so the whole traversal degrades instead of staying O(V+E).

The adjacency-list version (graph["km"] = [("m", 0.001)]) gives you that exact list in O(1) — graph[current] is the neighbor list, no scanning. That's the whole reason adjacency lists are the standard shape for any graph traversal (DFS, BFS, Dijkstra, all of them) — the algorithm's core operation is "get neighbors of current node," so the data structure should answer that directly instead of forcing a search every time.

"""

from collections import defaultdict

"""
conversion_factors = [["km", "m", 0.001], ["m", "cm", 0.01]]

1) km, m, 0.001

{"km": [(m, 0.001)], "m": [(km, 1000)]}

2) m, cm, 0.01

{"km": [(m, 0.001)], "m": [(km, 1000), (cm, 0.01)], "cm" : [(m, 100)]}


"""

def return_conversion(conversions: list[list], a: str, b: str) -> float:

    # Build graph
    graph = defaultdict(list)
    for unitA, unitB, factor in conversions:
        graph[unitA].append((unitB, factor))
        graph[unitB].append((unitA, 1/factor))

    if a not in graph or b not in graph:
        return -1

    # Go through graph
    def dfs(current, target, visited, product):
        if current == target:
            return product

        visited.add(current)

        for neighbor, factor in graph[current]:
            if neighbor not in visited:
                result = dfs(neighbor, target, visited, product * factor)   # recurse one hop further
                if result is not None:
                    return result            # found a path through this neighbor — stop searching, propagate up
                    
        return None                          # tried every neighbor, none led to target

    result = dfs(a, b, set(), 1)
    return result if result is not None else -1

"""
Complexity to state out loud:
- Time: O(V + E) — building the graph is O(E) (E = number of conversion triples, each adds 2 directed edges); DFS visits every unit at most once (guarded by `visited`) and walks every edge at most once, so V + E total.
- Space: O(V + E) for the adjacency list itself, plus O(V) for `visited` and O(V) worst-case recursion depth (a long chain of units strung together with no shortcut back to an already-visited node).
"""

conversion_factors = [["km", "m", 0.001], ["m", "cm", 0.01]]

print(return_conversion(conversion_factors, "km", "cm"))


# for unit1, unit2, factor in conversion_factors:
#     print(unit1, unit2, factor)

