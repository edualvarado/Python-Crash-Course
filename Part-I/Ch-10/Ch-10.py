from pathlib import Path
import json

path = Path('pi_digits.txt')
contents = path.read_text()
contents = contents.rstrip()

path = Path('files/example.txt')
contents = path.read_text()

# print(contents)

lines = contents.splitlines()
for line in lines:
    print(line)


pi_string = ''
for line in lines:
    pi_string += line.strip()

print(pi_string)

path = Path('files/writing.txt')

content = 'This is a new file.\n'
content += 'This is the second line.\n'
path.write_text(content)

# print('Give me two numbers, and I will divide them.')
# while True:
#     first_number = input("\nFirst number (or 'q' to quit): ")
#     if first_number == 'q':
#         break
#     second_number = input("Second number (or 'q' to quit): ")
#     if second_number == 'q':
#         break
#
#     try:
#         result = int(first_number) / int(second_number)
#     except ZeroDivisionError:
#         print("You can't divide by zero!")
#     except ValueError:
#         print("Please enter valid numbers.")
#     else:
#         print(f"The result is: {result}")

path = Path('files/exception.txt')
try:
    contents = path.read_text()
except FileNotFoundError:
    print(f"File {path} not found.")

numbers = [1, 2, 3, 4, 5]

path = Path('files/numbers.json')
contents = json.dumps(numbers)
path.write_text(contents)

contents = path.read_text()
numbers = json.loads(contents)

print(numbers)