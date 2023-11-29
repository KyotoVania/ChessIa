import random
import argparse
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessAI(nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



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
            value = piece_to_value.get(piece.piece_type, 0)
            if piece.color == chess.BLACK:
                value = -value
            board_state[0, i] = value

    return board_state


def choose_move_from_output(output, board):
    sorted_indices = torch.argsort(output, descending=True).view(-1)


    for i in range(0, len(sorted_indices), 2):
        from_square = sorted_indices[i].item()
        to_square = sorted_indices[i + 1].item() if i + 1 < len(sorted_indices) else -1
        move = chess.Move(from_square, to_square)
        if move in board.legal_moves:
            return move

    return random.choice(list(board.legal_moves))


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

        if move in board.legal_moves:
            board.push(move)
        else:
            print("Illegal move, try again." if mode == 'ai_vs_human' else "Invalid move selected.")
            continue
        if args.verbose:
            print(board)
            print("===========================================================================================================")

    print("===========================================================================================================")
    print(board)
    result = board.result(claim_draw=True)

    # Additional checks for game end conditions
    if result == "1/2-1/2" or result == "1-0" or result == "0-1":
        return result
    elif board.is_checkmate():
        return "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.can_claim_draw():
        return "1/2-1/2"
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return "1/2-1/2"
    else:
        return "1/2-1/2"



def run_continuous_games(ai_model, initial_mode='ai_vs_ai', verbose=False):
    mode = initial_mode
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    continue_running = True

    while continue_running:
        try:
            print("Starting new game in mode:", mode)
            result = play_chess_game(ai_model, mode=mode)
            results[result] += 1
            print(f"Game result: {result}. Total results: {results}")

        except KeyboardInterrupt:
            print("\nInterrupted!")
            print("Current Results:")
            print(results)
            print("\nChoose an option:")
            print("1: Continue")
            print("2: Change Mode")
            print("3: Exit")
            choice = input("Enter your choice (1, 2, or 3): ")

            if choice == '1':
                # Continue the loop
                continue
            elif choice == '2':
                new_mode = input("Enter new mode (ai_vs_human, ai_vs_ai, ai_vs_random): ")
                if new_mode in ['ai_vs_human', 'ai_vs_ai', 'ai_vs_random']:
                    mode = new_mode
                else:
                    print("Invalid mode. Please choose a valid mode.")
            elif choice == '3':
                continue_running = False
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")



# Main logic
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play Chess Game')
    parser.add_argument('--mode', type=str, default='ai_vs_human',
                        choices=['ai_vs_human', 'ai_vs_ai', 'ai_vs_random'],
                        help='Choose game mode: ai_vs_human, ai_vs_ai, or ai_vs_random')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuous games. If not set, run a single game.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output. If set, print the board after each move.')

    args = parser.parse_args()

    chess_model = ChessAI()

    if args.continuous:
        run_continuous_games(chess_model, initial_mode=args.mode, verbose=args.verbose)
    else:
        play_chess_game(chess_model, mode=args.mode)
