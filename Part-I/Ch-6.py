alien_0 = {"color": "green", "points": 5}
print(alien_0["color"])
print(alien_0["points"])
print(alien_0)

alien_0["x_position"] = 0
alien_0["y_position"] = 25

print(alien_0)

alien_0 = {}
alien_0["color"] = "red"
alien_0["points"] = 5
print(alien_0)

alien_0["color"] = "blue"
print(alien_0)

# del alien_0["points"]
# print(alien_0)

alien_speed = alien_0.get("speed", "No speed defined")
print(alien_speed)

user_0 = {
    "username": "efermi",
    "first": "Enrico",
    "last": "Fermi"}

for key, value in user_0.items():
    print(f"\nKey: {key}")
    print(f"Value: {value}")

# for key in user_0.keys():
#     print(f"\nKey: {key}")

favorite_languages = {
    "jen": "python",
    "sarah": "c",
    "edward": "ruby",
    "phil": "python"
}

friends = ["jen", "sarah", "marlon"]

for name in favorite_languages.keys():
    print(f"{name.title()}, thank you for taking the poll.")
    if name in friends:
        print(f"Hi {name.title()}, I see you love {favorite_languages[name].title()}!")

for name in friends:
    if name not in favorite_languages.keys():
        print(f"{name.title()}, please take our poll!")

for name in sorted(favorite_languages.keys()):
    print(f"{name.title()}, thank you for taking the poll.")

# ----

alien_0 = {"color": "green", "points": 5}
alien_1 = {"color": "yellow", "points": 10}
alien_2 = {"color": "red", "points": 15}

aliens = [alien_0, alien_1, alien_2]

for alien in aliens:
    print(alien)

aliens = []

for alien_number in range(30):
    new_alien = {"color": "green", "points": 5, "speed": "slow"}
    aliens.append(new_alien)

for alien in aliens[:5]:
    print(alien)
print(f"...{len(aliens)} aliens created.")

# ----

pizza = {
    "crust": "thick",
    "toppings": ["mushrooms", "extra cheese"]
}
print(f"You ordered a {pizza['crust']}-crust pizza with the following toppings: ")
for topping in pizza["toppings"]:
    print("\t" + topping)

favorite_languages = {
    "jen": ["python", "ruby"],
    "sarah": ["c"],
    "edward": ["ruby", "haskell"],
    "phil": ["python", "r"]
}

for name, languages in favorite_languages.items():
    print(f"\n{name.title()}'s favorite languages are:")
    for language in languages:
        print(f"\t{language.title()}")

# ----

users = {
    "aeinstein": {
        "first": "albert",
        "last": "einstein",
        "location": "princeton",
        "field": "physics"
    },
    "mcurie": {
        "first": "marie",
        "last": "curie",
        "location": "paris",
        "field": "chemistry"
    }
}

for username, user_info in users.items():
    print(f"\nUsername: {username}")
    full_name = f"{user_info['first']} {user_info['last']}"
    location = user_info["location"]
    field = user_info["field"]

    print(f"\tFull name: {full_name.title()}")
    print(f"\tLocation: {location.title()}")
    print(f"\tField: {field.title()}")