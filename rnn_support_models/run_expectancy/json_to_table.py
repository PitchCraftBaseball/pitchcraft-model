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
    default=30,
    help='Minimum occurrences threshold for coloring (default: 30)'
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
    p_pitch_types = ["FF", "SI", "SL", "CU", "FC", "CH", "ST", "KC", "FS", "SV", #"FA",
 "EP", "KN", #"CS",
 "FO", #"SC",
 "AB", "IN", #"PO",
 "UN"]
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
        #"FA": "Fastball",
        "EP": "Eephus",
        "KN": "Knuckleball",
        #"CS": "Slow Curve",
        "FO": "Forkball",
        #"SC": "Screwball",
        "AB": "Automatic Ball",
        "IN": "Intentional Ball",
        #"PO": "Pitchout",
        "UN": "Unknown",
    }

    pitch_classifications = {
        "FF": "fastball",
        "SI": "fastball",
        "FC": "fastball",
        #"FA": "fastball",

        "SL": "breaking",
        "ST": "breaking",
        "CU": "breaking",
        "KC": "breaking",
        "SV": "breaking",
        #"CS": "breaking",
        #"SC": "breaking",

        "CH": "offspeed",
        "FS": "offspeed",
        "FO": "offspeed",
        "KN": "offspeed",
        "EP": "offspeed",

        "AB": "other",
        "IN": "other",
        #"PO": "other",
        "UN": "other",
    }
    graphics_dir = f"graphics_{start_year}-{end_year}"
    os.makedirs(graphics_dir, exist_ok=True)
    
    # Create subdirectories for each pitch category
    categories = ["fastball", "breaking", "offspeed", "other"]
    for category in categories:
        category_dir = os.path.join(graphics_dir, category)
        os.makedirs(category_dir, exist_ok=True)
    
    print(f"Output directory: {graphics_dir}")

    # Generate heatmap for each pitch type
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

        # Convert to numpy arrays for visualization
        values = df.to_numpy(dtype=float)
        occurrences = df_occ.to_numpy(dtype=float)
        
        # Create mask for low occurrence cells
        low_occ_mask = occurrences <= min_occurrences
        
        # Calculate max value only from non-outlier (high occurrence) cells
        valid_values = values.copy()
        valid_values[low_occ_mask | np.isnan(values)] = np.nan
        max_value = np.nanmax(valid_values)

        if np.isnan(max_value) or max_value == 0:
            print(f"  Skipping {pitch_type} - insufficient data (max_value={max_value})")
            continue

        # Fixed bins at 0.25 intervals from 0 to 2.5+
        bins = [0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, np.inf]
        cmap = plt.get_cmap("Blues", len(bins) - 1)
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

        # Fixed threshold for text color (use white text for darker cells)
        threshold = 1.5
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                occ = occurrences[i, j]
                if np.isnan(value):
                    continue
                # For low occurrence cells, use black text on yellow background
                if low_occ_mask[i, j]:
                    text_color = "black"
                else:
                    text_color = "white" if value >= threshold else "black"
                # Format as "{run_expectancy}-{occurrence}"
                occ_display = int(occ) if not np.isnan(occ) else 0
                ax.text(j, i, f"{value:.3f}-{occ_display}", ha="center", va="center", color=text_color, fontsize=6)

        row_labels = [f"{outs}-{bases}" for outs, bases in df.index]
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")

        pitch_name = pitch_type_names.get(pitch_type, "Unknown Pitch")
        ax.set_title(f"{pitch_type} - {pitch_name} ({start_year}-{end_year})")

        fig.tight_layout()
        
        # Determine the category subdirectory for this pitch type
        category = pitch_classifications.get(pitch_type, "other")
        category_dir = os.path.join(graphics_dir, category)
        fig_path = os.path.join(category_dir, f"{pitch_type}_run_expectancy.png")
        
        fig.savefig(fig_path, dpi=200)
        print(f"  Generated {pitch_type} ({pitch_type_names.get(pitch_type, 'Unknown')}) -> {category}/{pitch_type}_run_expectancy.png")
        plt.close(fig)

    # Generate aggregated heatmaps for each category
    print("\nGenerating aggregated category heatmaps...")
    
    # Group pitch types by category
    category_pitch_types = {}
    for pitch_type in p_pitch_types:
        category = pitch_classifications.get(pitch_type, "other")
        if category not in category_pitch_types:
            category_pitch_types[category] = []
        category_pitch_types[category].append(pitch_type)
    
    for category, pitch_types_in_category in category_pitch_types.items():
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
                    # Aggregate data across all pitch types in this category
                    total_weighted_re = 0
                    total_occurrences = 0
                    
                    for pitch_type in pitch_types_in_category:
                        pitch_info = (
                            data
                            .get(outs, {})
                            .get(bases, {})
                            .get(count, {})
                            .get(pitch_type, {})
                        )
                        re = pitch_info.get("run_expectancy")
                        occ = pitch_info.get("occurrences")
                        
                        if re is not None and occ is not None:
                            total_weighted_re += re * occ
                            total_occurrences += occ
                    
                    # Calculate weighted average run expectancy
                    if total_occurrences > 0:
                        row[count] = total_weighted_re / total_occurrences
                        occ_row[count] = total_occurrences
                    else:
                        row[count] = None
                        occ_row[count] = None
                
                rows.append(row)
                occ_rows.append(occ_row)
        
        df = pd.DataFrame(rows)
        df = df.set_index(["outs", "bases"])
        df = df[p_counts]
        
        df_occ = pd.DataFrame(occ_rows)
        df_occ = df_occ.set_index(["outs", "bases"])
        df_occ = df_occ[p_counts]

        # Convert to numpy arrays for visualization
        values = df.to_numpy(dtype=float)
        occurrences = df_occ.to_numpy(dtype=float)
        
        # Create mask for low occurrence cells
        low_occ_mask = occurrences <= min_occurrences
        
        # Calculate max value only from non-outlier (high occurrence) cells
        valid_values = values.copy()
        valid_values[low_occ_mask | np.isnan(values)] = np.nan
        max_value = np.nanmax(valid_values)

        if np.isnan(max_value) or max_value == 0:
            print(f"  Skipping {category} - insufficient data (max_value={max_value})")
            continue

        # Fixed bins at 0.25 intervals from 0 to 2.5+
        bins = [0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, np.inf]
        cmap = plt.get_cmap("Blues", len(bins) - 1)
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

        # Fixed threshold for text color (use white text for darker cells)
        threshold = 1.5
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                occ = occurrences[i, j]
                if np.isnan(value):
                    continue
                # For low occurrence cells, use black text on yellow background
                if low_occ_mask[i, j]:
                    text_color = "black"
                else:
                    text_color = "white" if value >= threshold else "black"
                # Format as "{run_expectancy}-{occurrence}"
                # occ_display = int(occ) if not np.isnan(occ) else 0
                # ax.text(j, i, f"{value:.3f}-{occ_display}", ha="center", va="center", color=text_color, fontsize=6)

                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=6)

        row_labels = [f"{outs}-{bases}" for outs, bases in df.index]
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")

        category_name = category.capitalize()
        ax.set_title(f"{category_name} Pitches - Aggregated Run Expectancy ({start_year}-{end_year})")

        fig.tight_layout()
        
        category_dir = os.path.join(graphics_dir, category)
        fig_path = os.path.join(category_dir, f"{category}_run_expectancy.png")
        
        fig.savefig(fig_path, dpi=200)
        print(f"  Generated {category_name} aggregated -> {category}/{category}_run_expectancy.png")
        plt.close(fig)
