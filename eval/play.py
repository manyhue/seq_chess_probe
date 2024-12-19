import asyncio
import random
import chess
import ipywidgets as widgets
import torch
from lib.chess import iter_to_move_strings, chess_move_labels
from lib.utils import is_notebook

if is_notebook():
    from IPython.display import display, clear_output


class ChessGame:
    def __init__(self, model, le=chess_move_labels, _ai_illegal_attempts=5):
        """Initialize with a model and decoder"""
        self.model = model
        self.le = le
        self.device = next(model.parameters()).device
        self._ai_illegal_attempts = _ai_illegal_attempts
        self.model_name = model.filename().split("_")[0]
        self.notebook = False

    def draw_game_board(self, board):
        """Draw the current game board using chesslib"""
        if is_notebook():
            from IPython.display import display, clear_output

            boardsvg = chess.svg.board(board, size=350)
            clear_output(wait=True)
            display(boardsvg)
        else:
            print(board)

    def get_move_from_user(self, board):
        """Prompt the user for a move (if they're white) using widgets or questionary"""
        # if is_notebook():
        #     # Create a text input widget
        #     move_input_widget = widgets.Text(
        #         description="Enter move:", placeholder="e.g., Pe2e4"
        #     )
        #     # Create a button to submit the move
        #     submit_button = widgets.Button(description="Submit Move")

        #     # Display the widgets
        #     display(move_input_widget, submit_button)

        #     move_input = None

        #     def on_submit_button_click(b):
        #         """Submit move when button is clicked"""
        #         nonlocal move_input
        #         move_input = move_input_widget.value

        #     submit_button.on_click(on_submit_button_click)

        #     # Wait for the move to be entered and the button to be clicked
        #     while move_input is None:
        #         await asyncio.sleep(0.1)  # Async wait to prevent blocking

        # else:
        # Use questionary in non-notebook environment
        move_input = input("Enter your move (e.g., e2e4):")

        # Ensure the move is valid and push it to the board
        try:
            move = board.push_san(move_input)
        except ValueError:
            print("Invalid move! Please try again.")
            return self.get_move_from_user(board)

        return move

    # multinomial
    def model_prediction(self, board):
        move_strings = iter_to_move_strings(
            board.move_stack,
            seq_len=None,
            pad_token=self.le.inverse_transform([0])[0],
        )
        pred = self.model.generate(
            torch.tensor(self.le.transform(move_strings)).view(1, -1).to(self.device), 1
        ).cpu()  # Replace with actual model prediction

        decoded = self.le.inverse_transform(pred.flatten())[-1]
        return decoded

    def get_move_from_ai(self, board, max_tries=5):
        seen = set()
        move_str = None
        while len(seen) < max_tries:
            # Get the predicted move
            while move_str is None or move_str in seen:
                print("thinking...")
                move_str = self.model_prediction(board)
            try:
                # Try to make the move
                move = board.push_uci(move_str[1:])  # uci doesn't include the piece
                return move
            except ValueError:
                # If the move is illegal, retry with a new prediction
                seen.add(move_str)
                print(f"{self.model_name} attempted an illegal move: {move_str}.")

        # If all 5 attempts fail, resign
        print("All attempts failed. Resigning...")

        return None  # Resigning means no valid move was made

    # todo: refactor
    def check_game_status(board):
        if board.is_stalemate():
            return "Stalemate: The game is a draw due to no legal moves and no check."
        elif board.is_insufficient_material():
            return "Draw: Insufficient material to checkmate."
        elif board.is_seventyfive_moves():
            return "Draw: 75-move rule reached without progress."
        elif board.is_fivefold_repetition():
            return "Draw: Fivefold repetition rule reached."
        elif board.is_variant_draw():
            return "Draw: Variant-specific draw rule."
        elif board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            return f"Checkmate: {winner} wins!"
        else:
            return "Game is ongoing."

    def play(self, player_white=True, fen=None):
        """Main function to run the chess game with interaction and model prediction"""
        # Initialize the board with the provided FEN or default to the start position
        board = chess.Board(fen) if fen else chess.Board()

        if not player_white and board.turn:
            self.draw_game_board(board)
            board.push(random.choice(board.legal_moves))

        while not board.is_game_over():
            # Draw the board
            self.draw_game_board(board)

            move = self.get_move_from_user(board)
            print(f"You played: {move}")

            self.draw_game_board(board)

            # After making the move, ask the model for the next move
            move = self.get_move_from_ai(board, max_tries=self._ai_illegal_attempts)
            if move:
                print(f"{self.model_name} played: {move}")
            else:
                print(f"{self.model_name} resigned. You win!")
