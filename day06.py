from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from functools import total_ordering
from pathlib import Path


FILE = Path() / "docs" / "day07_test.txt"


class HandType(IntEnum):
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    FULL_HOUSE = 5
    FOUR_OF_A_KIND = 6
    FIVE_OF_A_KIND = 7


@dataclass
@total_ordering
class Hand:
    order = "23456789TJQKA"
    value: str
    hand_type: HandType

    def __init__(self, value: str, bet: int):
        self.value = value
        self.bet = bet

        c = Counter(self.value)
        if len(c) == 1:
            self.hand_type = HandType.FIVE_OF_A_KIND
        elif len(c) == 2:
            if max(c.values()) == 4:
                self.hand_type = HandType.FOUR_OF_A_KIND
            else:
                self.hand_type = HandType.FULL_HOUSE
        elif len(c) == 3:
            if max(c.values()) == 3:
                self.hand_type = HandType.THREE_OF_A_KIND
            else:
                self.hand_type = HandType.TWO_PAIR
        elif len(c) == 4:
            self.hand_type = HandType.ONE_PAIR
        else:
            self.hand_type = HandType.HIGH_CARD

    def to_tuple(self):
        return (self.hand_type,) + tuple(self.order.index(x) for x in self.value)

    def __eq__(self, value: "Hand") -> bool:
        return self.to_tuple() == value.to_tuple()

    def __lt__(self, value: "Hand") -> bool:
        return self.to_tuple() < value.to_tuple()


with open(FILE) as f:
    hands = []
    for line in f:
        hand, bet = line.rstrip("\n").split()
        hands.append(Hand(hand, int(bet)))

hands.sort()
winnings = 0
for idx, hand in enumerate(hands):
    winnings += (idx + 1) * hand.bet

print(f"1: {winnings}")
