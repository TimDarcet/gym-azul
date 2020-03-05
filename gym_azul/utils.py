from gym_azul.classes.tile import Tile


def tile_at(i, j):
    """Return the tile type at coordinates i, j of the square
    i is the y coordinate, j is the x coordinate, starting from the top left"""
    diff_to_tile = [
        Tile.BLUE,
        Tile.YELLOW,
        Tile.RED,
        Tile.BLACK,
        Tile.CYAN
    ]
    return diff_to_tile[(j - i) % 5]

def where_tile(l_idx, color):
    """Return where is the tile of color <color> on line <l_idx>"""
    return (l_idx + color.value) % len(Tile)

def print_side_by_side(*args):
    zip(*(a.split('\n') for a in args))
    #TODO