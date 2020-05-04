
from enum import Enum
class CarType(Enum):
    CROSSOVER = 1
    MUTATION = 2
    BEST = 3
    SAVED = 4
    RANDOM = 5
    USER = 6
    RL = 7

class Levels(Enum):
    LEVEL1 = 1
    LEVEL2 = 2

class GameType(Enum):
    GA = 1
    RL = 2