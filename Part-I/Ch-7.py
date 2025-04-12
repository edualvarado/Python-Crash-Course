message = input("Hello Python people! Enter text: ")
print(message)

# i = 0
# while i < 5:
#     print(i)
#     i += 1

# prompt = "\nTell me something, and I will repeat it back to you:"
# prompt += "\nEnter 'quit' to end the program."
#
# active = True
# while active:
#     message = input(prompt)
#     if message == 'quit':
#         active = False
#     else:
#         print(message)
#
# while True:
#     print("Please type 'quit' when you are done.")
#     message = input()
#     if message == 'quit':
#         break

# current_number = 1
# while current_number <= 10:
#     current_number += 1
#     if current_number % 2 == 0:
#         continue
#     print(current_number)

# Start with users that need to be verified

unverified_users = ['alice', 'brian', 'candace']
verified_users = []

while unverified_users:
    current_user = unverified_users.pop()
    print(f"Verifying user: {current_user.title()}")
    verified_users.append(current_user)

print("\nThe following users have been verified:")
for verified_user in verified_users:
    print(verified_user.title())

pets = ['dog', 'cat', 'dog', 'goldfish', 'cat', 'rabbit', 'cat']
print(pets)
while 'cat' in pets:
    pets.remove('cat')
print(pets)

responses = {}
polling = True
while polling:
    name = input("\nWhat is your name? ")
    responses[name] = input("Which mountain would you like to climb someday? ")
    repeat = input("Would you like to let another person respond? (yes/no) ")
    if repeat == 'no':
        polling = False
print("\n--- Poll Results ---")
for name, response in responses.items():
    print(f"{name} would like to climb {response}.")