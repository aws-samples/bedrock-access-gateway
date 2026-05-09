import time
import random

def calculate_factorial(n):
    if n == 0:
        return 1
    else:
        return n * calculate_factorial(n - 1)

def find_largest_number(numbers):
    largest = numbers[0]
    for num in numbers:
        if num > largest:
            largest = num
    return largest

def inefficient_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def print_user_info(self):
        print(f"Name: {self.name}, Age: {self.age}")

def process_data(data):
    result = []
    for item in data:
        if item % 2 == 0:
            result.append(item * 2)
        else:
            result.append(item * 3)
    return result

def generate_random_numbers(n):
    numbers = []
    for i in range(n):
        numbers.append(random.randint(1, 100))
    return numbers

def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    return average

def main():
    # Inefficient factorial calculation
    print(calculate_factorial(20))

    # Unnecessary loop for finding largest number
    numbers = [3, 7, 2, 9, 1, 5]
    print(find_largest_number(numbers))

    # Inefficient sorting algorithm
    unsorted_list = [64, 34, 25, 12, 22, 11, 90]
    print(inefficient_sort(unsorted_list))

    # Inconsistent naming convention
    user1 = User("John Doe", 30)
    user1.print_user_info()

    # Redundant if-else structure
    data = [1, 2, 3, 4, 5]
    print(process_data(data))

    # Inefficient random number generation
    random_numbers = generate_random_numbers(1000000)
    print(f"Generated {len(random_numbers)} random numbers")

    # Potential division by zero
    empty_list = []
    print(calculate_average(empty_list))

    # Unnecessary time delay
    time.sleep(5)
    print("Finished processing after 5 seconds")

if __name__ == "__main__":
    main()
