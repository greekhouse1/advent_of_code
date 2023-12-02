from pathlib import Path
import string


FILE = Path() / "docs" / "day1.txt"

DIGITS = set(string.digits)

def get_first(s: str) -> int:
    for letter in s:
        if letter in DIGITS:
            return int(letter)
    assert False, f"No digit found: {line}"

with open(FILE) as f:
    total = 0
    for line in f:
        first = get_first(line)
        last = get_first(line[::-1])
        total += 10*first + last

print(f"{total = }")

lu = {
"one": 1 , 
"two": 2 , 
"three" :3, 
"four" :4, 
"five" :5, 
"six" :6, 
"seven" :7, 
"eight" :8, 
"nine": 9
}