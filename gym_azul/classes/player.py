from gym_azul.utils import *

penalty_score = [0, -1, -2, -4, -6, -8, -11, -14]

class Player:
    def __init__(self):
        self.score = 0
        # TODO: What if we use here a dict Tile->bool ?
        # May be a better representation for the agent,
        # who will better understand the relationship between the queues and the square
        # But the point calculation will be less explicit though
        self.square = [[False] * 5 for _ in range(5)]
        self.queues = [[None, 0] for _ in range(5)]
        self.penalties = 0
    
    def end_round(self):
        # Update queue + square
        for i, (qt, ql) in enumerate(self.queues):
            assert 0 <= ql <= i + 1
            if ql == i + 1:
                # Put the tile on the square
                j = where_tile(i, qt)
                assert self.square[i][j] == False
                # Place the tile
                self.square[i][j] = True
                # Remove the tiles inqueue
                self.queues[i] = [None, 0]
                # Update score 
                self.update_score(i, j)
        # Update score for penalties
        self.score += penalty_score[self.penalties]
        # Remove penalties
        self.penalties = 0

    def observe(self):
        d = {
            "points": self.score,
            "square": tuple(list(map(int, self.square[i])) for i in range(5)),
            "queues": tuple({"type": q[0] if q[0] is None else q[0].value, "num": q[1]}
                            for q in self.queues),
            "penalties": self.penalties
        }
        return d

    def __str__(self):
        r = ''
        r += "Score: {}\n".format(self.score)
        r += "Square:\n" + '\n'.join(map(str, self.square)) + '\n'
        r += "Queues:\n{}\n".format(self.queues)
        r += "Penalties: {}\n".format(self.penalties)
        return r
    
    def update_score(self, i, j):
        """Update score after having placed tile (i, j)"""
        assert self.square[i][j], "Tile was not actually placed !"
        for k in range(5):
            # Count point added for tile i, k (horizontal tiles)
            a = min(j, k)
            b = max(j, k)
            self.score += int(all(self.square[i][l] for l in range(a, b + 1)))
        for k in range(5):
            # Count point added for tile k, j (vertical tiles)
            a = min(i, k)
            b = max(i, k)
            self.score += int(all(self.square[l][j] for l in range(a, b + 1)))
        # Special bonuses
        if all(self.square[i][k] for k in range(5)):
            print("Line completed")
            score += 2
        if all(self.square[k][j] for k in range(5)):
            print("Column completed")
            score += 7
        if all(self.square[k][where_tile(k, tile_at(i, j))] for k in range(5)):
            print("Color completed")
            score += 10
        
