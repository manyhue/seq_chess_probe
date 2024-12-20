#!/usr/bin/env zsh

hfdownloader -d ezipe/lichess_elo_binned_debug

for dir in */; do
    d=$(basename "$dir")
    if [[ $d =~ ^[0-9]+$ ]]; then
        zst_file=("$dir"*.zst)
        echo $zst_file
        if [[ $d -ge 400 && $d -le 1800 ]]; then
            mkdir -p low
            output_file="low/${d}.pgn"
            zstd -d "$zst_file" -o "$output_file"
        elif [[ $d -ge 1900 && $d -le 2400 ]]; then
            mkdir -p medium
            output_file="medium/${d}.pgn"
            zstd -d "$zst_file" -o "$output_file"
        elif [[ $d -ge 2500 && $d -le 3200 ]]; then
            mkdir -p high
            output_file="high/${d}.pgn"
            zstd -d "$zst_file" -o "$output_file"
        elif [[ $d -ge 3200 && $d -le 4000 ]]; then
            mkdir -p top
            output_file="top/${d}.pgn"
            zstd -d "$zst_file" -o "$output_file"
        fi

        # Now duplicate every newline in the decompressed file
        tmp_file="${output_file}.tmp"
        while IFS= read -r line; do
            echo "$line" >>"$tmp_file"
            echo "" >>"$tmp_file" # Duplicate newline
        done <"$output_file"

        # Replace the original file with the duplicated newline version
        mv "$tmp_file" "$output_file"
        # rm $d # or do manually
    fi
done
