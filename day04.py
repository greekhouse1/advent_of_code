from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


FILE = Path() / "docs" / "day04.txt"


def parse_line(s: str) -> Tuple[List, List]:
    _, rest = s.split(":")
    s1, s2 = rest.split("|")
    winners = [int(x) for x in s1.split()]
    my_numbers = [int(x) for x in s2.split()]
    return winners, my_numbers


def score_card(winners: List[int], numbers: List[int]) -> int:
    matched = 0
    for n in numbers:
        if n in winners:
            matched += 1

    if not matched:
        return 0

    return 2 ** (matched - 1)


total1 = 0

with open(FILE) as f:
    for line in f:
        winners, my_numbers = parse_line(line.rstrip("\n"))
        total1 += score_card(winners, my_numbers)

print(f"{total1 = }")

total2 = 0

card_counts = defaultdict(lambda: 1)


def score_card(winners: List[int], numbers: List[int]) -> int:
    matched = 0
    for n in numbers:
        if n in winners:
            matched += 1

    return matched


with open(FILE) as f:
    for idx, line in enumerate(f):
        total2 += card_counts[idx]
        winners, my_numbers = parse_line(line.rstrip("\n"))
        num_winners = score_card(winners, my_numbers)
        for i in range(num_winners):
            card_counts[idx + i + 1] += card_counts[idx]

print(f"{total2 = }")
