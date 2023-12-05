from dataclasses import dataclass
from pathlib import Path
from typing import List

FILE = Path() / "docs" / "day02.txt"


@dataclass
class Pull:
    red: int
    green: int
    blue: int

    def power(self):
        return self.red * self.blue * self.green

    def __eq__(self, value: "Pull") -> bool:
        return (
            self.red == value.red
            and self.green == value.green
            and self.blue == value.blue
        )

    def __le__(self, value: "Pull") -> bool:
        return (
            self.red <= value.red and self.green <= value.green and self.blue <= value.blue
        )

    @classmethod
    def from_string(cls, s: str) -> "Pull":
        red, green, blue = 0, 0, 0
        for cubes in s.split(", "):
            count, color = cubes.split()
            if color == "red":
                red = int(count)
            elif color == "green":
                green = int(count)
            elif color == "blue":
                blue = int(count)
        return Pull(red, green, blue)


@dataclass
class Game:
    id: int
    pulls: List[Pull]

    def get_min_bag(self) -> Pull:
        red = max(x.red for x in self.pulls)
        green = max(x.green for x in self.pulls)
        blue = max(x.blue for x in self.pulls)
        return Pull(red, green, blue)


    @classmethod
    def from_string(cls, s: str) -> "Game":
        id_info, game_info = s.split(":")
        id = int(id_info.split(" ")[-1])
        pulls = [Pull.from_string(x) for x in game_info.split(";")]
        return Game(id=id, pulls=pulls)


TARGET = Pull(red=12, green=13, blue=14)

test = Game.from_string("Game 98: 3 red, 13 green, 7 blue")
print(f"{test = }")
print(f" {all(x <= TARGET for x in test.pulls)}")

with open(FILE) as f:
    total = 0
    for line in f:
        game = Game.from_string(line.rstrip("\n"))
        if all(x <= TARGET for x in game.pulls):
            total += game.id


print(f"{total = }")

with open(FILE) as f:
    total = 0
    for line in f:
        game = Game.from_string(line.rstrip("\n"))
        min_bag = game.get_min_bag()
        print
        total += min_bag.power()
print(f"{total = }")

