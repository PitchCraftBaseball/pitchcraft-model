"""
Run Expectancy 288 Analysis by Pitch Type

This script calculates run expectancy for each pitch type across all 288 game situations
(3 outs x 8 base states x 12 counts). Data is pulled directly from the AWS database.

Output: JSON file with structure {outs: {bases: {count: {pitch_type: {occurrences, run_expectancy}}}}}
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path  
from typing import Dict  
import pandas as pd

# Add parent directory to path to import model_shared
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_shared.db import get_read_cursor
from model_shared.logger import logger

START_YEAR = 2015
END_YEAR = 2025

# Pitch description categorization
categorize_description = {
    'hit_into_play': 'P',
    'foul': 'F',
    'ball': 'B',
    'foul_tip': 'S',
    'swinging_strike': 'S',
    'swinging_strike_blocked': 'S',
    'called_strike': 'S',
    'foul_bunt': 'S',
    'blocked_ball': 'B',
    'hit_by_pitch': 'HBP',
    'missed_bunt': 'S',
    'pitchout': 'X',
    'bunt_foul_tip': 'S',
}

categorize_description = defaultdict(lambda: 'X', categorize_description)

# Possible outs, counts, and base situations
P_OUTS = [0, 1, 2]
P_COUNTS = ['00', '01', '02', '10', '11', '12', '20', '21', '22', '30', '31', '32']
P_BASES = ['XXX', 'OXX', 'XOX', 'OOX', 'XXO', 'OXO', 'XOO', 'OOO']


def generate_count(balls: int, strikes: int) -> str:
    """Generate count string from balls and strikes."""
    return str(balls) + str(strikes)


def generate_inning_code(game_pk: int, inning: int, inning_topbot: str) -> str:
    """Generate unique inning identifier."""
    return str(game_pk) + str(inning) + str(inning_topbot)


def situation_to_identifier(
    outs: int,
    count: str,
    on_1b: bool,
    on_2b: bool,
    on_3b: bool
) -> str:
    """Convert game situation to identifier string (e.g., '011OXX')."""
    output = str(outs) + count
    for runner in [on_1b, on_2b, on_3b]:
        output += 'O' if runner else 'X'
    return output


def fetch_pitch_data_from_db(start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    """
    Fetch all pitch data from the database for the specified year range.
    
    Filters:
    - Excludes Exhibition (E) and Spring Training (S) games only
    - Includes Regular season (R) + Playoffs (F/D/L/W) to match original script behavior
    - Removes NULL and 'ABS' pitch types
    
    Args:
        start_year: First year to include (inclusive)
        end_year: Last year to include (inclusive)
    
    Returns:
        DataFrame with all pitch data needed for run expectancy calculation
    """
    logger.info(f"Fetching pitch data from database for years {start_year}-{end_year}...")
    
    query = """
        SELECT 
            game_pk,
            inning,
            inning_topbot,
            at_bat_number,
            balls,
            strikes,
            outs_when_up,
            on_1b,
            on_2b,
            on_3b,
            bat_score,
            post_bat_score,
            pitch_type,
            description,
            game_date
        FROM historical_pitches
        WHERE game_year >= %s 
          AND game_year <= %s
          AND game_type NOT IN ('E', 'S')
          AND pitch_type IS NOT NULL
          AND pitch_type != 'ABS'
        ORDER BY game_pk, inning, CASE WHEN inning_topbot = 'Top' THEN 0 ELSE 1 END, at_bat_number, pitch_number
    """
    
    with get_read_cursor() as cursor:
        cursor.execute(query, (start_year, end_year))
        
        # Fetch all rows
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        logger.info(f"Fetched {len(rows):,} pitch records from database")
        
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the pitch data to add computed columns.
    
    Args:
        df: Raw pitch data from database
    
    Returns:
        DataFrame with additional computed columns
    """
    logger.info("Preprocessing pitch data...")
    
    # Generate count string
    df['count'] = df.apply(lambda row: generate_count(row['balls'], row['strikes']), axis=1)
    
    # Generate inning codes
    logger.info("Generating inning codes...")
    df['inning_code'] = df.apply(
        lambda row: generate_inning_code(row['game_pk'], row['inning'], row['inning_topbot']),
        axis=1
    )
    
    # Calculate runs scored in each inning
    logger.info("Calculating runs scored per inning...")
    inning_runs = {}
    for inning_code, group in df.groupby('inning_code'):
        # Get the final score of the inning (last at-bat's post_bat_score)
        final_score = group.sort_values('at_bat_number', ascending=False).iloc[0]['post_bat_score']
        inning_runs[inning_code] = final_score
    
    df['post_inn_score'] = df['inning_code'].map(inning_runs)
    df['runs_to_score'] = df['post_inn_score'] - df['bat_score']
    
    logger.info("Calculating base runner situations...")
    # Convert base runner IDs to boolean presence
    df['rofirst'] = df['on_1b'].notna() & (df['on_1b'] != 0)
    df['rosecond'] = df['on_2b'].notna() & (df['on_2b'] != 0)
    df['rothird'] = df['on_3b'].notna() & (df['on_3b'] != 0)
    
    # Generate situation identifier
    df['situation_identifier'] = df.apply(
        lambda row: situation_to_identifier(
            row['outs_when_up'],
            row['count'],
            row['rofirst'],
            row['rosecond'],
            row['rothird']
        ),
        axis=1
    )
    
    # Categorize pitch descriptions
    df['description_cat'] = df['description'].map(categorize_description)
    
    logger.info("Preprocessing complete.")
    return df


def calculate_re288_by_pitch_type(df: pd.DataFrame) -> Dict:
    """
    Calculate run expectancy for each pitch type across all 288 situations.
    
    Args:
        df: Preprocessed pitch data
    
    Returns:
        Nested dictionary with run expectancy data
    """
    logger.info("Calculating run expectancy by pitch type for all 288 situations...")
    
    pitch_types = sorted(df['pitch_type'].dropna().unique().tolist())
    logger.info(f"Found {len(pitch_types)} pitch types: {pitch_types}")
    
    json_output = {}
    
    for outs in P_OUTS:
        json_output[outs] = {}
        for bases in P_BASES:
            json_output[outs][bases] = {}
            for count in P_COUNTS:
                count_format = count[0] + '-' + count[1]
                key = str(outs) + count + bases
                
                # Filter to this specific situation
                view = df[df['situation_identifier'] == key]
                
                pitch_entries = []
                for pitch_type in pitch_types:
                    pitch_view = view[view['pitch_type'] == pitch_type]
                    occurrences = len(pitch_view)
                    
                    if occurrences > 0:
                        run_expectancy = round(pitch_view['runs_to_score'].mean(), 3)
                        pitch_entries.append((pitch_type, int(occurrences), run_expectancy))
                
                # Sort by occurrences (most common first)
                pitch_entries.sort(key=lambda x: x[1], reverse=True)
                
                json_output[outs][bases][count_format] = {}
                for pitch_type, occurrences, run_expectancy in pitch_entries:
                    json_output[outs][bases][count_format][pitch_type] = {
                        'occurrences': occurrences,
                        'run_expectancy': run_expectancy
                    }
    
    logger.info("Run expectancy calculation complete.")
    return json_output


def main(start_year: int = START_YEAR, end_year: int = END_YEAR, output_dir: Path | None = None):
    """
    Main execution function.
    
    Args:
        start_year: First year to include in analysis
        end_year: Last year to include in analysis
        output_dir: Directory to save output file (defaults to analysis/)
    """
    try:
        # Fetch data from database
        df = fetch_pitch_data_from_db(start_year, end_year)
        
        # Preprocess
        df = preprocess_data(df)
        
        # Calculate run expectancy
        json_output = calculate_re288_by_pitch_type(df)
        
        # Save to file
        if output_dir is None:
            output_dir = Path(__file__).parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = output_dir / f'{start_year}-{end_year}_re288_pitch.json'
        
        with open(output_filename, 'w') as file:
            json.dump(json_output, file, indent=2)
        
        logger.info(f"✓ {output_filename} saved to disk.")
        print(f"✓ Successfully generated {output_filename}")
        print(f"  - Years: {start_year}-{end_year}")
        print(f"  - Total pitches analyzed: {len(df):,}")
        print(f"  - Output location: {output_filename.absolute()}")
        
    except Exception as e:
        logger.error(f"Error during RE288 calculation: {e}", exc_info=True)
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate run expectancy by pitch type for all 288 game situations"
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=START_YEAR,
        help=f'First year to include (default: {START_YEAR})'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=END_YEAR,
        help=f'Last year to include (default: {END_YEAR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for JSON file (default: analysis/)'
    )
    
    args = parser.parse_args()
    
    main(
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir
    )
