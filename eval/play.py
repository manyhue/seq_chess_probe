import asyncio
import random
from typing import Callable, List, Optional, Any
import chess
import ipywidgets as widgets
import torch
from lib.chess import iter_to_move_strings, chess_move_labels
from lib.utils import dbg, is_notebook

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

    @classmethod
    def get_move_from_user(cls, board):
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
            if move == "":
                print("Null move entered. Resigning...")
                return None
            print("Invalid move! Please try again.")
            return cls.get_move_from_user(board)

        return move

    # multinomial
    def model_prediction(self, board, model):
        move_strings = iter_to_move_strings(
            board.move_stack,
            seq_len=None,
            pad_token=self.le.inverse_transform([0])[0],
        )
        pred = model.generate(
            torch.tensor(self.le.transform(move_strings)).view(1, -1).to(self.device), 1
        ).cpu()  # Replace with actual model prediction

        decoded = self.le.inverse_transform(pred.flatten())[-1]
        return decoded

    def get_move_from_ai(
        self, board, max_tries=5, max_samples=5, move_hist=False, model=None
    ):
        if model is None:
            model = self.model
            model_name = self.model_name
        else:
            model_name = model.filename().split("_")[0]
        seen = set()
        move_str = None
        while len(seen) < max_tries:
            count = 0
            # Get the predicted move
            while move_str is None or move_str in seen:
                if count < max_samples:
                    print("thinking...")
                    move_str = self.model_prediction(board, model)
                    count += 1
                else:
                    print("Generation of multiple legal moves failed. Resigning...")
                    return None
            try:
                # Try to make the move
                move = board.push_uci(move_str[1:])  # uci doesn't include the piece
                if isinstance(move_hist, list):
                    seen.add(move)
                    move_hist.append(seen)
                return move
            except ValueError:
                # If the move is illegal, retry with a new prediction
                seen.add(move_str)
                print(f"{model_name} attempted an illegal move: {move_str}.")

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

    def play(
        self,
        player_white=True,
        self_move_fn: Optional[Callable[[chess.Board], Any]] = None,  # type: ignore
        fen=None,
        verbose=False,
        return_board=False,
    ):
        def get_outcome():
            if (outcome := board.outcome()) is not None:
                return outcome
            nonlocal game_over
            return (
                chess.Termination.VARIANT_WIN
                if game_over == 1
                else chess.Termination.VARIANT_LOSS
                if game_over == -1
                else chess.Termination.VARIANT_DRAW
            )

        game_over = None
        """Main function to run the chess game with interaction and model prediction"""
        # Initialize the board with the provided FEN or default to the start position
        board = chess.Board(fen) if fen else chess.Board()
        their_move_hist = []

        if not callable(self_move_fn):
            self_move_fn = self.get_move_from_user
        elif player_white:
            self.draw_game_board(board)
            board.push(random.choice(list(board.legal_moves)))  # initial move for model

        if not player_white:  # todo: handle fen
            self.draw_game_board(board)
            board.push(random.choice(list(board.legal_moves)))

        while board.outcome() is None and game_over is None:
            # Draw the board
            self.draw_game_board(board)

            move = self_move_fn(board)
            if move is not None:
                print(f"You played: {move}")
            else:
                print("You resigned.")
                game_over = -1
                if verbose:
                    print(their_move_hist)
                break

            self.draw_game_board(board)

            # After making the move, ask the model for the next move
            move = self.get_move_from_ai(
                board,
                max_tries=self._ai_illegal_attempts,
                move_hist=False if not verbose else their_move_hist,
            )
            if move is not None:
                print(f"{self.model_name} played: {move}")
            else:
                print(f"{self.model_name} resigned.")
                game_over = 1
                if verbose:
                    print(their_move_hist)

        print(board.outcome(), game_over)

        if return_board:
            return get_outcome(), board
        return get_outcome()
