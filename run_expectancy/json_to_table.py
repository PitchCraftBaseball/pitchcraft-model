# open {start_year}-{end_year}_re288_pitch.json and convert to a table with columns for situation identifier, pitch type, occurrences, and run expectancy

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import argparse

# Default year range
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2025

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Convert RE288 pitch JSON to tables and heatmaps"
)
parser.add_argument(
    '--start-year',
    type=int,
    default=DEFAULT_START_YEAR,
    help=f'First year to include (default: {DEFAULT_START_YEAR})'
)
parser.add_argument(
    '--end-year',
    type=int,
    default=DEFAULT_END_YEAR,
    help=f'Last year to include (default: {DEFAULT_END_YEAR})'
)
parser.add_argument(
    '--min-occurrences',
    type=int,
    default=10,
    help='Minimum occurrences threshold for coloring (default: 10)'
)

args = parser.parse_args()

start_year = args.start_year
end_year = args.end_year
min_occurrences = args.min_occurrences

file_name = f'{start_year}-{end_year}_re288_pitch.json'
print(f"Processing {file_name}...")

with open(file_name, 'r') as file:
    data = json.load(file)
    p_outs = ['0', '1', '2']
    p_bases = ['XXX', 'OXX', 'XOX', 'XXO', 'OOX', 'OXO', 'XOO', 'OOO']
    p_counts = ["0-2", "1-2", "0-1", "2-2", "1-1", "0-0", "1-0", "2-1", "3-2", "2-0", "3-1", "3-0"]
    p_pitch_types = ["FF", "SI", "SL", "CU", "FC", "CH", "ST", "KC", "FS", "SV", "FA", "EP", "KN", "CS", "FO", "SC"]
    pitch_type_names = {
        "FF": "Four-Seam Fastball",
        "SI": "Sinker",
        "SL": "Slider",
        "CU": "Curveball",
        "FC": "Cutter",
        "CH": "Changeup",
        "ST": "Sweeper",
        "KC": "Knuckle Curve",
        "FS": "Splitter",
        "SV": "Slurve",
        "FA": "Fastball",
        "EP": "Eephus",
        "KN": "Knuckleball",
        "CS": "Slow Curve",
        "FO": "Forkball",
        "SC": "Screwball",
    }

    output_dir = f"pitch_type_tables_{start_year}-{end_year}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # generate a table for each pitch type where column are counts, and rows are bases and outs, and the values are run expectancy
    for pitch_type in p_pitch_types:
        rows = []
        occ_rows = []
        for outs in p_outs:
            for bases in p_bases:
                row = {
                    "outs": outs,
                    "bases": bases,
                }
                occ_row = {
                    "outs": outs,
                    "bases": bases,
                }
                for count in p_counts:
                    pitch_info = (
                        data
                        .get(outs, {})
                        .get(bases, {})
                        .get(count, {})
                        .get(pitch_type, {})
                    )
                    row[count] = pitch_info.get("run_expectancy")
                    occ_row[count] = pitch_info.get("occurrences")
                rows.append(row)
                occ_rows.append(occ_row)

        df = pd.DataFrame(rows)
        df = df.set_index(["outs", "bases"])
        df = df[p_counts]
        
        df_occ = pd.DataFrame(occ_rows)
        df_occ = df_occ.set_index(["outs", "bases"])
        df_occ = df_occ[p_counts]

        output_path = os.path.join(output_dir, f"{pitch_type}_run_expectancy.csv")
        df.to_csv(output_path)
        
        occ_path = os.path.join(output_dir, f"{pitch_type}_occurrences.csv")
        df_occ.to_csv(occ_path)

    graphics_dir = os.path.join(output_dir, "graphics")
    os.makedirs(graphics_dir, exist_ok=True)

    for file_name in os.listdir(output_dir):
        if not file_name.endswith("_run_expectancy.csv"):
            continue

        csv_path = os.path.join(output_dir, file_name)
        df = pd.read_csv(csv_path, index_col=[0, 1])
        values = df.to_numpy(dtype=float)
        
        # Load occurrences
        occ_file = file_name.replace("_run_expectancy.csv", "_occurrences.csv")
        occ_path = os.path.join(output_dir, occ_file)
        df_occ = pd.read_csv(occ_path, index_col=[0, 1])
        occurrences = df_occ.to_numpy(dtype=float)
        
        # Create mask for low occurrence cells
        low_occ_mask = occurrences <= min_occurrences
        
        # Calculate max value only from non-outlier (high occurrence) cells
        valid_values = values.copy()
        valid_values[low_occ_mask | np.isnan(values)] = np.nan
        max_value = np.nanmax(valid_values)

        if np.isnan(max_value) or max_value == 0:
            continue

        bins = np.linspace(0, max_value, 11)
        cmap = plt.get_cmap("Blues", 10)
        cmap.set_bad(color="#f0f0f0")
        norm = BoundaryNorm(bins, cmap.N, clip=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create a custom colored array
        colored_values = np.full_like(values, np.nan)
        # For high occurrence cells, use actual values for blue gradient
        colored_values[~low_occ_mask] = values[~low_occ_mask]
        
        masked = np.ma.masked_invalid(colored_values)
        im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
        
        # Overlay yellow for low occurrence cells
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if low_occ_mask[i, j] and not np.isnan(values[i, j]):
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                              facecolor='yellow', edgecolor='none'))

        threshold = max_value * 0.6
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if np.isnan(value):
                    continue
                # For low occurrence cells, use black text on yellow background
                if low_occ_mask[i, j]:
                    text_color = "black"
                else:
                    text_color = "white" if value >= threshold else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=7)

        row_labels = [f"{outs}-{bases}" for outs, bases in df.index]
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")

        pitch_code = file_name.replace("_run_expectancy.csv", "")
        pitch_name = pitch_type_names.get(pitch_code, "Unknown Pitch")
        ax.set_title(f"{pitch_code} - {pitch_name} ({start_year}-{end_year})")

        fig.tight_layout()
        image_name = file_name.replace(".csv", ".png")
        fig_path = os.path.join(graphics_dir, image_name)
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
