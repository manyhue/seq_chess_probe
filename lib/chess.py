import itertools
import chess
import chess.pgn
import io

from lib.utils import dbg, vb

# pgn = chess.pgn.read_game(io.StringIO(pgn_string)) -> pgn.mainline_moves()
from sklearn.preprocessing import LabelEncoder


def iter_to_move_strings(moves, seq_len, pad_token="<PAD>"):
    board = chess.Board()

    move_strings = []

    # Iterate through each move in the PGN game
    for move in itertools.islice(moves, 0, seq_len):
        if move is None or (board.piece_at(move.from_square) is None):
            print("Invalid move", board, move, move.from_square)
            break
        start_square = chess.square_name(move.from_square)
        end_square = chess.square_name(move.to_square)

        if move.promotion:
            promotion_piece = {
                chess.QUEEN: "Q",
                chess.ROOK: "R",
                chess.BISHOP: "B",
                chess.KNIGHT: "N",
            }[move.promotion]
            move_string = f"{board.piece_at(move.from_square).symbol()}{start_square}{end_square}={promotion_piece}"
        else:
            move_string = (
                f"{board.piece_at(move.from_square).symbol()}{start_square}{end_square}"
            )

        move_strings.append(move_string)

        board.push(move)
    if seq_len is not None and len(move_strings) < seq_len:
        move_strings.extend([pad_token] * (seq_len - len(move_strings)))

    return move_strings


def parse_pgn_files_to_move_strings(pgn_files, seq_len, pad_token="<PAD>"):
    """
    Parse a sequence of PGN files and generate move strings for each game.

    Args:
        pgn_files (list): List of file paths to PGN files.
        seq_len (int): Maximum sequence length of moves.
        pad_token (str): Token used to pad the sequences.

    Yields:
        list: A list of move strings for each game, padded to seq_len.
    """

    for pgn_file in pgn_files:
        with open(pgn_file, "r") as pgn:
            if vb(5):
                print("\nopening:", pgn_file)
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                if "FEN" in game.headers:
                    continue
                yield iter_to_move_strings(
                    game.mainline_moves(), seq_len, pad_token
                )  # todo: what sort of exceptions to handle?


def generate_chess_moves():
    # Define board files and ranks
    files = "abcdefgh"
    ranks = "12345678"

    # Store all generated moves
    moves = []

    # Helper function to check if a square is valid
    def is_valid_square(square):
        return len(square) == 2 and square[0] in files and square[1] in ranks

    # Pawn moves
    def generate_pawn_moves():
        pawn_moves = []

        # White pawns
        for file in files:
            # Standard forward moves
            for start_rank in "234567":
                end_rank = str(int(start_rank) + 1)
                pawn_moves.append(f"P{file}{start_rank}{file}{end_rank}")

                # Two-square moves from 2nd rank
                if start_rank == "2":
                    pawn_moves.append(f"P{file}2{file}4")

                # Diagonal captures
                for capture_file in [chr(ord(file) - 1), chr(ord(file) + 1)]:
                    if capture_file in files:
                        pawn_moves.append(
                            f"P{file}{start_rank}{capture_file}{end_rank}"
                        )

            # Promotions for white pawns
            for capture_file in [chr(ord(file) - 1), file, chr(ord(file) + 1)]:
                if capture_file in files:
                    pawn_moves.extend(
                        [
                            f"P{file}7{capture_file}8=Q",
                            f"P{file}7{capture_file}8=R",
                            f"P{file}7{capture_file}8=B",
                            f"P{file}7{capture_file}8=N",
                        ]
                    )

        # Black pawns (similar logic, but moving down the board)
        for file in files:
            # Standard forward moves
            for start_rank in "765432":
                end_rank = str(int(start_rank) - 1)
                pawn_moves.append(f"p{file}{start_rank}{file}{end_rank}")

                # Two-square moves from 7th rank
                if start_rank == "7":
                    pawn_moves.append(f"p{file}7{file}5")

                # Diagonal captures
                for capture_file in [chr(ord(file) - 1), chr(ord(file) + 1)]:
                    if capture_file in files:
                        pawn_moves.append(
                            f"p{file}{start_rank}{capture_file}{end_rank}"
                        )

            for capture_file in [chr(ord(file) - 1), file, chr(ord(file) + 1)]:
                if capture_file in files:
                    pawn_moves.extend(
                        [
                            f"p{file}2{capture_file}1=Q",
                            f"p{file}2{capture_file}1=R",
                            f"p{file}2{capture_file}1=B",
                            f"p{file}2{capture_file}1=N",
                        ]
                    )

        return pawn_moves

    def generate_knight_moves():
        knight_moves = []
        knight_deltas = [
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
            (2, 1),
            (2, -1),
            (-2, 1),
            (-2, -1),
        ]

        for start_file in files:
            for start_rank in ranks:
                for dx, dy in knight_deltas:
                    end_file = chr(ord(start_file) + dx)
                    end_rank = str(int(start_rank) + dy)

                    # White and black knights
                    if is_valid_square(f"{end_file}{end_rank}"):
                        knight_moves.append(
                            f"n{start_file}{start_rank}{end_file}{end_rank}"
                        )
                        knight_moves.append(
                            f"N{start_file}{start_rank}{end_file}{end_rank}"
                        )

        return knight_moves

    def generate_king_moves():
        king_moves = []
        king_deltas = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        for start_file in files:
            for start_rank in ranks:
                for dx, dy in king_deltas:
                    end_file = chr(ord(start_file) + dx)
                    end_rank = str(int(start_rank) + dy)

                    # White and black kings
                    if is_valid_square(f"{end_file}{end_rank}"):
                        king_moves.append(
                            f"k{start_file}{start_rank}{end_file}{end_rank}"
                        )
                        king_moves.append(
                            f"K{start_file}{start_rank}{end_file}{end_rank}"
                        )

        return king_moves

    def generate_rook_moves():
        rook_moves = []

        for start_file in files:
            for start_rank in ranks:
                # Horizontal moves
                for end_file in files:
                    if end_file != start_file:
                        rook_moves.append(
                            f"r{start_file}{start_rank}{end_file}{start_rank}"
                        )
                        rook_moves.append(
                            f"R{start_file}{start_rank}{end_file}{start_rank}"
                        )

                # Vertical moves
                for end_rank in ranks:
                    if end_rank != start_rank:
                        rook_moves.append(
                            f"r{start_file}{start_rank}{start_file}{end_rank}"
                        )
                        rook_moves.append(
                            f"R{start_file}{start_rank}{start_file}{end_rank}"
                        )

        return rook_moves

    def generate_bishop_moves():
        bishop_moves = []

        for start_file in files:
            for start_rank in ranks:
                for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    current_file, current_rank = start_file, start_rank
                    while True:
                        current_file = chr(ord(current_file) + dx)
                        current_rank = str(int(current_rank) + dy)

                        if is_valid_square(f"{current_file}{current_rank}"):
                            bishop_moves.append(
                                f"b{start_file}{start_rank}{current_file}{current_rank}"
                            )
                            bishop_moves.append(
                                f"B{start_file}{start_rank}{current_file}{current_rank}"
                            )
                        else:
                            break

        return bishop_moves

    def generate_queen_moves():
        queen_moves = []

        for start_file in files:
            for start_rank in ranks:
                # Horizontal and vertical moves
                for end_file in files:
                    if end_file != start_file:
                        queen_moves.append(
                            f"q{start_file}{start_rank}{end_file}{start_rank}"
                        )
                        queen_moves.append(
                            f"Q{start_file}{start_rank}{end_file}{start_rank}"
                        )

                for end_rank in ranks:
                    if end_rank != start_rank:
                        queen_moves.append(
                            f"q{start_file}{start_rank}{start_file}{end_rank}"
                        )
                        queen_moves.append(
                            f"Q{start_file}{start_rank}{start_file}{end_rank}"
                        )

                # Diagonal moves
                for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    current_file, current_rank = start_file, start_rank
                    while True:
                        current_file = chr(ord(current_file) + dx)
                        current_rank = str(int(current_rank) + dy)

                        if is_valid_square(f"{current_file}{current_rank}"):
                            queen_moves.append(
                                f"q{start_file}{start_rank}{current_file}{current_rank}"
                            )
                            queen_moves.append(
                                f"Q{start_file}{start_rank}{current_file}{current_rank}"
                            )
                        else:
                            break

        return queen_moves

    def generate_castling_moves():
        return [
            # White castling
            "Ke1g1",  # Kingside castling
            "Ke1c1",  # Queenside castling
            # Black castling
            "ke8g8",  # Kingside castling
            "ke8c8",  # Queenside castling
        ]

    # Modify existing method to include castling moves
    moves.extend(generate_castling_moves())

    # Combine all moves
    moves.extend(generate_pawn_moves())
    moves.extend(generate_knight_moves())
    moves.extend(generate_king_moves())
    moves.extend(generate_rook_moves())
    moves.extend(generate_bishop_moves())
    moves.extend(generate_queen_moves())
    moves.extend(generate_castling_moves())

    return moves


def board_to_model_input(board, seq_len, pad_token="<PAD>"):
    """Converts a list of moves (from move_stack) into a list of move strings based on the given board state."""
    move_strings = []

    # Iterate through each move in the board's move history (move_stack)
    for move in board.move_stack[:seq_len]:
        start_square = chess.square_name(move.from_square)
        end_square = chess.square_name(move.to_square)

        if move.promotion:
            promotion_piece = {
                chess.QUEEN: "Q",
                chess.ROOK: "R",
                chess.BISHOP: "B",
                chess.KNIGHT: "N",
            }[move.promotion]
            move_string = f"{board.piece_at(move.from_square).symbol()}{start_square}{end_square}={promotion_piece}"
        else:
            move_string = (
                f"{board.piece_at(move.from_square).symbol()}{start_square}{end_square}"
            )

        move_strings.append(move_string)

    # Pad the move_strings list if necessary
    if len(move_strings) < seq_len:
        move_strings.extend([pad_token] * (seq_len - len(move_strings)))

    return move_strings


chess_move_labels = LabelEncoder()
chess_move_labels.fit(["<PAD>"] + generate_chess_moves())
