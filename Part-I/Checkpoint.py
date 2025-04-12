A = [1, 2, 3]
B = [4, 5, 6, 1]

for i in A:
    print(f"\tA: {i}")
    if i in B:
        print(f"{i} in B")

element = B.pop()
print(f'Element popped out {element}')

for i in A:
    print(f"\tA: {i}")
    if i in B:
        print(f"{i} in B")

for i in B:
    A.append(i)

print(f"len(A) = {len(A)}")
print(f"A = {A}")

series = []

for i in range(len(A)):
    print(f"\ti: {i}")
    series.append(i**2)

print(f"series = {series}")

series = [value ** 2 for value in range(8)]
print(f"series = {series}")

tuple = (1, 2, 3)
print(f"tuple = {tuple}")

planets = {
    "Earth" : {
        "Type": "Planet",
        "Location": {"X": 1, "Y": 15}
    },
    "Mars" : {
        "Type": "Exo",
        "Location": {"X": 3, "Y": 1}
    }
}

for planet, info in planets.items():
    print(f"\nPlanet: {planet}")
    print(f"\tType: {info['Type']}")
    print(f"\tLocation: {info['Location']['X']}, {info['Location']['Y']}")



