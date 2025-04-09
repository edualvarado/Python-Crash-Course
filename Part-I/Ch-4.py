list = ['a', 'b', 'c', 'd', 'e']
for i in list:
    print(i)

for i in range(len(list)):
    print(i)
    print(list[i])

for i in range(1, 5, 2):
    print(i)

squares = []
for i in range(1,6):
    squares.append(i**2)
print(squares)

print(min(squares))
print(max(squares))
print(sum(squares))

squares = [value ** 2 for value in range(1, 10)]
print(squares)

print(squares[0:3])
print(squares[:5])
print(squares[5:])
print(squares[-2:])
print(squares[7:])

new_squares = squares[:]
print(new_squares)

dimensions = (200, 50)  # tuple = immutable list
print(dimensions[0])
print(dimensions[1])

for dimension in dimensions:
    print(dimension)

dimensions = (400, 100)
for dimension in dimensions:
    print(dimension)