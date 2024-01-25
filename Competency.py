# Databricks notebook source
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC %md
# MAGIC # Control Flow 

# COMMAND ----------

# Handling division by zero
numerator = 10
denominator = 0

try:
    result = numerator / denominator
except ZeroDivisionError:
    print("Error: Cannot divide by zero.")
else:
    print("Result:", result)
finally:
    print("This block always executes.")


# COMMAND ----------

# Break the loop when a specific condition is met
numbers = [1, 2, 3, 4, 5, 6]

for num in numbers:
    if num == 4:
        print("Number found! Breaking the loop.")
        break
    print(num)


# COMMAND ----------

# Skip the iteration when a specific condition is met
numbers = [2, 4, 6, 7, 8, 9]
for num in numbers:
    if num % 2 == 0:
        print("Even number. Skipping.")
        continue
    print(num)


# COMMAND ----------

# MAGIC %md
# MAGIC #  Variable Type 

# COMMAND ----------

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Create an instance of the Person class
person1 = Person("Bob", 30)

# Retrieve data from a class instance using attributes
person_name = person1.name
print("Person Name:", person_name)


# COMMAND ----------

def example_function():
    x = 10  # This is a local variable
    print(x)

example_function()
# print(x)  # This would result in an error since x is not defined outside the function

# COMMAND ----------

y = 20  # This is a global variable

def another_function():
    global y  # Using the global keyword to access the global variable
    print(y)

another_function()
print(y)  # This would print the global variable y
