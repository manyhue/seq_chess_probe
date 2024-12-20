def split_pgn(input_file, games_per_file=100):
    with open(input_file, 'r', encoding='utf-8') as f:
        pgn_data = f.read()

    # Split the PGN data into individual games based on the "[Event" string
    games = pgn_data.split("[Event")

    # Skip the first part since it's before the first game
    games = games[1:]

    # Calculate the number of files needed
    num_files = len(games) // games_per_file + (1 if len(games) % games_per_file != 0 else 0)

    for i in range(num_files):
        start = i * games_per_file
        end = min((i + 1) * games_per_file, len(games))

        # Prepare the games for the current file, including the "[Event" part
        chunk = "[Event".join(games[start:end])

        output_file = f"game_part_{i + 1}.pgn"
        
        # Write the chunk of games to a new file
        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.write("[Event" + chunk)  # Add "[Event" at the beginning

# Usage
split_pgn("megadb.pgn")
