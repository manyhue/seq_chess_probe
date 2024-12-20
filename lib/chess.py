import itertools
import random
import chess
import chess.pgn
import io

from lib.utils import dbg, vb

# pgn = chess.pgn.read_game(io.StringIO(pgn_string)) -> pgn.mainline_moves()
from sklearn.preprocessing import LabelEncoder

def movetext_to_moves(movetext):
    return chess.pgn.read_game(io.StringIO(movetext)).mainline_moves()

def iter_to_move_strings(moves, seq_len, pad_token="<PAD>", return_board=False):
    board = chess.Board()

    move_strings = []

    # Iterate through each move in the PGN game
    for move in itertools.islice(moves, 0, seq_len):
        if move is None or (board.piece_at(move.from_square) is None):
            print("Invalid move", board, move, move.from_square)
            break
            # apparently a1a1, move= 0000 is accepted somehow?
        start_square = chess.square_name(move.from_square)
        end_square = chess.square_name(move.to_square)

        move_string = (
            f"{board.piece_at(move.from_square).symbol()}{start_square}{end_square}"
        )

        if move.promotion:
            promotion_piece = {
                chess.QUEEN: "q",
                chess.ROOK: "r",
                chess.BISHOP: "b",
                chess.KNIGHT: "n",
            }[move.promotion]
            move_string += promotion_piece

        move_strings.append(move_string)

        board.push(move)
    if seq_len is not None and len(move_strings) < seq_len:
        move_strings.extend([pad_token] * (seq_len - len(move_strings)))

    if return_board:
        return move_strings, board
    return move_strings


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
                            f"P{file}7{capture_file}8q",
                            f"P{file}7{capture_file}8r",
                            f"P{file}7{capture_file}8b",
                            f"P{file}7{capture_file}8n",
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
                            f"p{file}2{capture_file}1q",
                            f"p{file}2{capture_file}1r",
                            f"p{file}2{capture_file}1b",
                            f"p{file}2{capture_file}1n",
                        ]
                    )

        return pawn_moves

    # corrected version
    # def generate_pawn_moves():
    #     pawn_moves = []

    #     # White pawns
    #     for file in files:
    #         # Standard forward moves
    #         for start_rank in "234567":
    #             end_rank = str(int(start_rank) + 1)

    #             # Promotions for white pawns
    #             for capture_file in [chr(ord(file) - 1), file, chr(ord(file) + 1)]:
    #                 if capture_file in files:
    #                     if start_rank == "7":
    #                         pawn_moves.extend(
    #                             [
    #                                 f"P{file}7{capture_file}8q",
    #                                 f"P{file}7{capture_file}8r",
    #                                 f"P{file}7{capture_file}8b",
    #                                 f"P{file}7{capture_file}8n",
    #                             ]
    #                         )
    #                     else:
    #                         pawn_moves.append(
    #                             f"P{file}{start_rank}{capture_file}{end_rank}"
    #                         )

    #             # Two-square moves from 2nd rank
    #             if start_rank == "2":
    #                 pawn_moves.append(f"P{file}2{file}4")

    #     # Black pawns (similar logic, but moving down the board)
    #     for file in files:
    #         # Standard forward moves
    #         for start_rank in "765432":
    #             end_rank = str(int(start_rank) - 1)
    #             # Promotions for white pawns
    #             for capture_file in [chr(ord(file) - 1), file, chr(ord(file) + 1)]:
    #                 if capture_file in files:
    #                     if start_rank == "2":
    #                         pawn_moves.extend(
    #                             [
    #                                 f"p{file}2{capture_file}1q",
    #                                 f"p{file}2{capture_file}1r",
    #                                 f"p{file}2{capture_file}1b",
    #                                 f"p{file}2{capture_file}1n",
    #                             ]
    #                         )
    #                     else:
    #                         pawn_moves.append(
    #                             f"p{file}{start_rank}{capture_file}{end_rank}"
    #                         )

    #             # Two-square moves from 2nd rank
    #             if start_rank == "7":
    #                 pawn_moves.append(f"p{file}2{file}4")

    #     return pawn_moves

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

        move_string = (
            f"{board.piece_at(move.from_square).symbol()}{start_square}{end_square}"
        )

        if move.promotion:
            promotion_piece = {
                chess.QUEEN: "q",
                chess.ROOK: "r",
                chess.BISHOP: "b",
                chess.KNIGHT: "n",
            }[move.promotion]
            move_string += promotion_piece

        move_strings.append(move_string)

    # Pad the move_strings list if necessary
    if len(move_strings) < seq_len:
        move_strings.extend([pad_token] * (seq_len - len(move_strings)))

    return move_strings


chess_move_labels = LabelEncoder()
chess_move_labels.fit(["<PAD>"] + generate_chess_moves())

ep_squares = [
    "a3",
    "b3",
    "c3",
    "d3",
    "e3",
    "f3",
    "g3",
    "h3",  # 3rd row
    "a6",
    "b6",
    "c6",
    "d6",
    "e6",
    "f6",
    "g6",
    "h6",  # 6th row
    "-",
]

fen_labels = LabelEncoder()
fen_labels.fit(
    [
        "K",  # also castling indicators
        "Q",
        "R",
        "B",
        "N",
        "P",  # White pieces
        "k",
        "q",
        "r",
        "b",
        "n",
        "p",  # Black pieces
        "z",  # Empty squares (run length decode from digits)
        "/",  # Row separator
        " ",  # Space (separates fields in FEN)
        "black",
        "white",
    ]
    + ep_squares
)


def generate_random_game(length):
    """
    Generates a random chess game of the given length.

    Args:
        length (int): The number of moves to generate.

    Returns:
        chess.Board: The final board state after the random moves.
        list: A list of moves in UCI format representing the game.
    """
    board = chess.Board()
    moves = []

    for _ in range(length):
        legal_moves = list(board.legal_moves)
        if (
            not legal_moves
        ):  # Check if there are no legal moves (checkmate or stalemate)
            break

        move = random.choice(legal_moves)
        board.push(move)
        moves.append(move.uci())

    return board


def generate_random_games_to_pgn(n_files, games_per_file=100, game_length=20):
    """
    Generates random chess games and saves them to PGN files.

    Args:
        games_per_file (int): Number of games per file.
        n_files (int): Number of files to create.
        game_length (int): Number of moves in each game.

    Saves:
        PGN files with games in the format: "games_<index>.pgn"
    """
    for file_index in range(n_files):
        file_name = f"games_{file_index + 1}.pgn"
        with open(file_name, "a") as pgn_file:
            for _ in range(games_per_file):
                board = generate_random_game(game_length)

                # Convert the board to a PGN game
                game = chess.pgn.Game()
                node = game
                for move in board.move_stack:
                    node = node.add_variation(move)

                # Write the game to the PGN file
                print(game, file=pgn_file, end="\n\n")

        print(f"Saved {games_per_file} games to {file_name}")


# for debugging
def moves_to_pgn(moves):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    if isinstance(moves[0], int):
        moves = chess_move_labels.inverse_transform(moves)

    for move in moves:
        uci_move = move[1:].lower()  # Convert to lowercase and strip promotion notation
        if board.parse_uci(uci_move):
            node = node.add_variation(board.push_uci(uci_move))
    game.headers["Event"] = "?"
    game.headers["Site"] = "?"
    game.headers["Date"] = "????.??.??"
    game.headers["Round"] = "?"
    game.headers["White"] = "?"
    game.headers["Black"] = "?"
    game.headers["Result"] = "*"
    print(str(game))
