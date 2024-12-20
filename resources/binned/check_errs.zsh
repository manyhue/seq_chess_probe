#!/bin/zsh

for file in **/*.pgn; do
echo $file
  python3 -c "
import chess.pgn
with open('$file', 'r') as pgn:
    chess.pgn.read_game(pgn)
    next_game = chess.pgn.read_game(pgn)
    if next_game is None:
        print('$file', 'err')
  "
done
