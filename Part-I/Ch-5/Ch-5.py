cars = ['suzuki', 'toyota', 'honda', 'nissan', 'ford']

for car in cars:
    if car == 'toyota':
        print(car.upper())
if 'toyota' in cars:
    print('Toyota is in the list')

if 'chevrolet' not in cars:
    print('Chevrolet is not in the list')

if 'chevrolet' in cars:
    print('Chevrolet is in the list')
elif 'toyota' in cars:
    print('Toyota is not in the list')
else:
    print('What?')

empty_list = []

if empty_list:
    print('List is not empty')
else:
    print('List is empty')