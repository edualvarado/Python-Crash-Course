# import cars
from cars import *

my_tesla = Car('tesla', 'model s', 2016)

print(my_tesla.get_descriptive_name())

my_tesla.battery.describe_battery()

my_tesla.battery.get_range()

my_tesla.battery.upgrade_battery(222)

my_tesla.battery.get_range()

my_tesla.update_odometer(2322)

my_tesla.read_odometer()

my_nissan = ElectricCar('nissan', 'skyline', 2016)

print(my_nissan.get_descriptive_name())
my_nissan.battery.describe_battery()
my_nissan.battery.get_range()
my_nissan.fill_gas_tank()
