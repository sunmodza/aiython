def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


numbers = [fibonacci(i) for i in range(11)]
assert numbers[-1] == 55


explanation: str = explain the pattern in numbers in English

print(numbers)
print(explanation)
