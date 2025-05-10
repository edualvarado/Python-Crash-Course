def greed_user(username, pet = "fish"):
    print(f"Hello World {username}, {pet}")

greed_user("Edu", "Dog")
greed_user("Other", "Cat")
greed_user("Ahh")


def get_formatted_name(first_name, last_name):
    full_name = f"{first_name} {last_name}"
    return full_name.title()

musician = get_formatted_name("jimi", "hendrix")
print(musician)

def get_formatted_name(first_name, last_name, middle_name = ""):
    if middle_name:
        full_name = f"{first_name} {middle_name} {last_name}"
    else:
        full_name = f"{first_name} {last_name}"

    return full_name.title()

musician = get_formatted_name("jimi", "hendrix", "sd")
print(musician)

def build_person(first_name, last_name, age = ''):
    person = {'first': first_name, 'last': last_name}
    if age:
        person['age'] = age
    return person

musician = build_person('jimi', 'hendrix', 27)
print(musician)

def greet_users(users):
    for user in users:
        print(f"Hello {user.title()}")

users = ['hannah', 'ty', 'margot']
greet_users(users)

unprinted_designs = ['iphone case', 'robot pendant', 'dodecahedron']
completed_models = []

# while unprinted_designs:
#     current_design = unprinted_designs.pop()
#     print(f"Printing model: {current_design}")
#     completed_models.append(current_design)
#
# print("\nThe following models have been printed:")
# for completed_model in completed_models:
#     print(completed_model)

def print_models(unprinted_designs, completed_models):
    while unprinted_designs:
        current_design = unprinted_designs.pop()
        print(f"Printing model: {current_design}")
        completed_models.append(current_design)

def show_completed_models(completed_models):
    print("\nThe following models have been printed:")
    for completed_model in completed_models:
        print(completed_model)

# Here, we send a copy of the list!
print_models(unprinted_designs[:], completed_models)
show_completed_models(completed_models)

def make_pizza(size, *args):
    print(f"\nMaking a {size}-inch pizza with the following toppings:")
    for topping in args:
        print(f"- {topping}")

make_pizza(16, 'pepperoni')

make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')

def build_profile(first, last, **kwargs):
    profile = {}
    profile['first_name'] = first
    profile['last_name'] = last
    for key, value in kwargs.items():
        profile[key] = value
    return profile

user_profile = build_profile('albert', 'einstein', location = 'princeton', field = 'physics')
print(user_profile)