from enum import IntEnum, auto, unique

@unique
class Tile(IntEnum):
    __order__ = "BLUE YELLOW RED BLACK CYAN"
    BLUE = auto()
    YELLOW = auto()
    RED = auto()
    BLACK = auto()
    CYAN = auto()