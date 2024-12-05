from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from functools import total_ordering
from pathlib import Path


FILE = Path() / "docs" / "day07.txt"


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
    order = "J23456789TQKA"
    value: str
    hand_type: HandType

    def __init__(self, value: str, bet: int):
        self.value = value
        self.bet = bet

        if "J" not in self.value:
            self.hand_type = self.get_hand_type(Counter(self.value))
        elif self.value == "JJJJJ":
            self.hand_type = HandType.FIVE_OF_A_KIND
        else:
            possible_hand_types = []
            non_joker = [x for x in self.value if x != "J"]
            for x in non_joker:
                new_value = self.value.replace("J", x)
                possible_hand_types.append(self.get_hand_type(Counter(new_value)))
                self.hand_type = max(possible_hand_types)


    def get_hand_type(self, c: Counter):
        if len(c) == 1:
            return HandType.FIVE_OF_A_KIND
        elif len(c) == 2:
            if max(c.values()) == 4:
                return HandType.FOUR_OF_A_KIND
            else:
                return HandType.FULL_HOUSE
        elif len(c) == 3:
            if max(c.values()) == 3:
                return HandType.THREE_OF_A_KIND
            else:
                return HandType.TWO_PAIR
        elif len(c) == 4:
            return HandType.ONE_PAIR
        else:
            return HandType.HIGH_CARD

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

print(f"2: {winnings}")
