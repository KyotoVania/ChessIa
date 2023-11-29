import random
import chess

board = chess.Board()

for _ in range(10):
    legal_moves = list(board.legal_moves)
    random_move = random.choice(legal_moves)
    board.push(random_move)
    print(board)