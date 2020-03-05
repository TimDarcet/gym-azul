from enum import IntEnum, auto, unique

@unique
class Tile(IntEnum):
    __order__ = "BLUE YELLOW RED BLACK CYAN"
    BLUE = 0
    YELLOW = 1
    RED = 2
    BLACK = 3
    CYAN = 4