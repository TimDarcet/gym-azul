class Player:
    def __init__(self):
        self.victory_points = 0
        self.square = [[False] * 5 for _ in range(5)]
        self.queues = [[0, None] for _ in range(5)]
        self.penalties = 0
