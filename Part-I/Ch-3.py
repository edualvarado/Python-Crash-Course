my_list = ['a', 'b', 'c', 'd', 'e']
print(my_list)
print(my_list[0])
print(my_list[-1])

my_list[0] = 'z'
print(my_list)

my_list.append('f')
print(my_list)

my_list.insert(2, 'new')
print(my_list)


my_removed_element = my_list.pop()
print(my_list)
print(my_removed_element)

my_removed_element = my_list.pop(2)
print(my_list)
print(my_removed_element)

my_list.remove('z')
print(my_list)

my_other_removed_element = 'b'
my_list.remove(my_other_removed_element)
print(my_list)
print(my_other_removed_element)

my_list = ['b', 'b', 'e', 'a', 'c']
print(my_list)
my_list.sort()
print(my_list)
my_list.sort(reverse=True)
print(my_list)

my_list = ['b', 'b', 'e', 'a', 'c']
print(my_list)
print(sorted(my_list))

my_list = ['b', 'b', 'e', 'a', 'c']
print(my_list)
my_list.reverse()
print(my_list)

len(my_list)




