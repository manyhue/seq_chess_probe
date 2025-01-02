import random
from typing import Callable, List, Optional, Any, Union
import chess
from lib.chess import iter_to_moves, chess_move_labels, moves_to_torch
from tnibs.utils import Base, dbg, is_notebook, vb
from tnibs.modules import Module

# just for names
class ChessPlayer(Base):
    def __init__(
        self, player: Union[None, Module], name=None, max_illegal=5, max_samples=5
    ):
        self.save_attr()
        if name is None:
            if player is None:  # user
                self.name = ""
            else:
                self.name = player.filename().split("_")[0]
        else:
            self.name = name

        if player is not None:
            self._device = next(player.parameters()).device

        self.illegal_history = []

    def get_move(self, board):
        return (
            self.get_move_from_user(board)
            if self.player is None
            else self.get_move_from_ai(board)
        )

    # san format
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

        # Ensure the move is valid and push it to the self.board
        for _ in range(self.max_illegal):
            try:
                move_input = input("Enter your move (e.g., e2e4):")

                if move_input.lower() == "r":
                    print("Resigning...")
                    return None

                move = board.parse_san(move_input)
                break
            except ValueError:
                print("Invalid move! Please try again.")
        else:
            print("All attempts failed. Resigning...")

        return move

    def get_move_from_ai(self, board):
        seen = set()
        move_str = None
        move = None
        while len(seen) < self.max_illegal:
            count = 0
            # Get the predicted move
            while move_str is None or move_str in seen:
                if count < self.max_samples:
                    if vb(7):
                        print("thinking...")
                    moves = iter_to_moves(
                        board.move_stack,
                        seq_len=None,
                    )
                    inputs = moves_to_torch(moves).view(1, -1).to(self._device)
                    move_str = self._model_prediction(self.player, inputs)
                    count += 1
                else:
                    print("Generation of multiple legal moves failed. Resigning...")
                    self.illegal_history.append(seen)
                    return None
            try:
                # Try to make the move
                move = board.parse_san(move_str[1:])  # uci doesn't include the piece
                break
            except ValueError:
                # If the move is illegal, retry with a new prediction
                seen.add(move_str)
                if vb(6):
                    print(f"{self.name} attempted an illegal move: {move_str}.")
        else:
            print("All attempts failed. Resigning...")

        self.illegal_history.append(seen)
        return move

    # multinomial
    @staticmethod
    def _model_prediction(model, inputs):
        pred = model.generate(inputs, 1).cpu()  # Replace with actual model prediction

        decoded = chess_move_labels.inverse_transform(pred.flatten())[-1]
        return decoded


class ChessGame:
    def __init__(
        self,
        p2_model,
        p1_model=None,
        p1_max_illegal=5,
        p2_max_illegal=5,
        draw_board=None,
    ):
        """Initialize with a model and decoder"""
        self.players = [
            ChessPlayer(p1_model, max_illegal=p1_max_illegal),
            ChessPlayer(p2_model, max_illegal=p2_max_illegal),
        ]

        if self.players[0].name == "":
            if self.players[1].name == "":
                self.players[0].name = "P1"
            else:
                self.players[0].name = "You"
        if self.players[1].name == "":
            self.players[1].name = "P2"
        if self.players[1].name == self.players[0].name:
            self.players[1].name += "(1)"

        self.notebook = False
        self.board = None

        self.draw_board = (
            draw_board
            if isinstance(draw_board, str)
            else "svg"
            if is_notebook()
            else "print"
        )

    def draw_game_board(self):
        """Draw the current game self.board using chesslib"""
        if self.draw_board == "svg":
            from IPython.display import display, clear_output

            boardsvg = chess.svg.board(self.board, size=350)
            clear_output(wait=True)
            display(boardsvg)
        elif self.draw_board == "print":
            print(self.board)

    @property
    def player_to_move(self):
        return (self.turn - (1 if self.p1_white else 0)) % 2

    def get_move(self):
        player = self.players[self.player_to_move]
        move = player.get_move(self.board)
        if move is None:
            self.game_over = -1
        else:
            self.board.push(move)

    # todo: refactor
    def check_game_status(self):
        board = self.board
        if board.is_stalemate():
            return "Stalemate: The game is a draw due to no legal moves and no check."
        elif board.is_insufficient_material():
            return "Draw: Insufficient material to checkmate."
        elif board.is_seventyfive_moves():
            return "Draw: 75-move rule reached without progress."
        elif board.is_fivefold_repetition():
            return "Draw: Fivefold repetition rule reached."  # autodraw whereas 3fold must be claimed
        elif board.is_variant_draw():
            return "Draw: Variant-specific draw rule."
        elif board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            return f"Checkmate: {winner} wins!"
        else:
            return "Game is ongoing."

    # returns an easy to understand outcome
    def get_outcome(self):
        result = (
            chess.Termination.VARIANT_DRAW
            if self.game_over == 0
            else chess.Termination.VARIANT_WIN  # use variant_win to denote side who wins
            if self.game_over is not None
            else outcome.termination
            if (outcome := self.board.outcome())
            else None  # None for continuing
        )
        if result is None:
            return None
        elif result in (chess.Termination.VARIANT_WIN, chess.Termination.CHECKMATE):
            return (
                result.name,
                (-1 if self.player_to_move == 0 else 1),
            )  # +1 if P1 wins
        else:
            return (result.name, 0)

    def init_game(self, fen=None, p1_white=True):
        if fen == "":
            self.board = chess.Board()
        elif fen is None:
            self.board = self.board or chess.Board()
        else:
            self.board = chess.Board(fen)
        self.game_over = None
        self.turn = 1
        self.p1_white = p1_white

    # 1 indexed
    def set_player(self, model, player=1, max_illegal=None):
        self.player[player - 1] = ChessPlayer(
            model,
            max_illegal=max_illegal or self.player[player - 1].max_illegal,
        )

    def play(self, p1_white=True, retain=True, fen=""):
        """Main function to run the chess game with interaction and model prediction. See init_game for effect of fen."""
        # Initialize the self.board with the provided FEN or default to the start position
        self.init_game(fen=fen, p1_white=p1_white)

        if p1_white:
            model_starts = self.players[0].player is not None
        else:
            model_starts = self.players[1].player is not None

        # initial move if model start
        if model_starts:
            self.draw_game_board()
            self.board.push(random.choice(list(self.board.legal_moves)))

        while (outcome := self.get_outcome()) is None:
            # Draw the self.board
            self.draw_game_board()
            self.get_move()
            self.turn += 1

        # print(self.board.outcome(), self.game_over)

        if not retain:
            self.board = None

        return outcome
