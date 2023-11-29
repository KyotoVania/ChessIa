import random
import argparse
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessAI(nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        # Définir les couches ici
        self.fc1 = nn.Linear(64, 128)  # 64 entrées pour chaque case, 128 neurones dans la couche cachée
        self.fc2 = nn.Linear(128, 64)  # 64 sorties pour chaque case

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Example function to convert board state to model input
def board_to_input(board):
    piece_to_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6
    }

    board_state = torch.zeros(1, 64)
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            # Assign a value based on piece type and color
            value = piece_to_value.get(piece.piece_type, 0)
            if piece.color == chess.BLACK:
                value = -value
            board_state[0, i] = value

    return board_state


# Example function to choose a move from model's output
def choose_move_from_output(output, board):
    # Flatten the output and sort it
    sorted_indices = torch.argsort(output, descending=True).view(-1)

    # Iterate over sorted indices to find a legal move
    for i in range(0, len(sorted_indices), 2):
        from_square = sorted_indices[i].item()
        to_square = sorted_indices[i + 1].item() if i + 1 < len(sorted_indices) else -1
        move = chess.Move(from_square, to_square)
        if move in board.legal_moves:
            return move

    # If no legal move found, return a random legal move
    return random.choice(list(board.legal_moves))


# Main function to play a game
def play_chess_game(ai_model, mode='ai_vs_human'):
    board = chess.Board()
    while not board.is_game_over():
        input_data = board_to_input(board)
        output = ai_model(input_data)

        if mode == 'ai_vs_human':
            if board.turn:  # AI's turn
                move = choose_move_from_output(output, board)
            else:  # Human's turn
                print("Enter your move (e.g., 'e2e4'):")
                print("Legal moves:", [move.uci() for move in board.legal_moves])
                try:
                    human_move = input()
                    move = chess.Move.from_uci(human_move)
                except ValueError:
                    print("Invalid input, try again.")
                    continue

        elif mode == 'ai_vs_ai':
            move = choose_move_from_output(output, board)

        elif mode == 'ai_vs_random':
            if board.turn:  # AI's turn
                move = choose_move_from_output(output, board)
            else:  # Random's turn
                move = random.choice(list(board.legal_moves))

        # Check if the move is legal and execute it
        if move in board.legal_moves:
            board.push(move)
        else:
            print("Illegal move, try again." if mode == 'ai_vs_human' else "Invalid move selected.")
            continue

        print(board)

    print("===========================================================================================================")
    print("Game over.")
    print("Reason:", board.result())
    print(board)


# Example usage
chess_model = ChessAI()

parser = argparse.ArgumentParser(description='Play Chess Game')
parser.add_argument('--mode', type=str, default='ai_vs_human',
                    choices=['ai_vs_human', 'ai_vs_ai', 'ai_vs_random'],
                    help='Choose game mode: ai_vs_human, ai_vs_ai, or ai_vs_random')
args = parser.parse_args()

# Instantiate the model
chess_model = ChessAI()

play_chess_game(chess_model, mode=args.mode)
