"""
Pitch Arsenal JSON Generator

Builds a pitcher-indexed JSON from AWS historical pitch data.

Current output step:
{
  "<pitcher_id>": {
	"player_name": "<name>",
	"pitch_type": {
	  "FF": 123,
	  "SL": 88
	}
  }
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to import model_shared
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_shared.db import get_read_cursor
from model_shared.logger import logger

START_YEAR = 2015
END_YEAR = 2025

pitch_classifications = {
	"FF": "fastball",
	"SI": "fastball",
	"FC": "fastball",
	# "FA": "fastball",

	"SL": "breaking",
	"ST": "breaking",
	"CU": "breaking",
	"KC": "breaking",
	"SV": "breaking",
	# "CS": "breaking",
	# "SC": "breaking",

	"CH": "offspeed",
	"FS": "offspeed",
	"FO": "offspeed",
	"KN": "offspeed",
	"EP": "offspeed",

	"AB": "other",
	"IN": "other",
	# "PO": "other",
	"UN": "other",
}

active_pitch_types = tuple(pitch_classifications.keys())


def fetch_pitcher_pitch_type_rows(start_year: int = START_YEAR, end_year: int = END_YEAR) -> list[tuple[int, str, str, int, int]]:
	"""
	Fetch pitcher rows with canonical player name and pitch type occurrence counts.

	Each returned row is:
	(pitcher_id, canonical_player_name, pitch_type, occurrences, pitcher_total_rows)
	"""
	logger.info(f"Fetching pitcher pitch-type usage for years {start_year}-{end_year}...")

	query = """
		WITH pitcher_name_counts AS (
			SELECT
				pitcher,
				player_name,
				COUNT(*) AS name_occurrences,
				ROW_NUMBER() OVER (
					PARTITION BY pitcher
					ORDER BY COUNT(*) DESC, player_name ASC
				) AS rn
			FROM historical_pitches
			WHERE game_year >= %s
			  AND game_year <= %s
			  AND game_type NOT IN ('E', 'S')
			  AND pitcher IS NOT NULL
			  AND player_name IS NOT NULL
			GROUP BY pitcher, player_name
		), canonical_names AS (
			SELECT pitcher, player_name
			FROM pitcher_name_counts
			WHERE rn = 1
		), pitch_type_counts AS (
			SELECT
				pitcher,
				pitch_type,
				COUNT(*) AS occurrences
			FROM historical_pitches
			WHERE game_year >= %s
			  AND game_year <= %s
			  AND game_type NOT IN ('E', 'S')
			  AND pitcher IS NOT NULL
			  AND pitch_type IS NOT NULL
			  AND pitch_type = ANY(%s)
			GROUP BY pitcher, pitch_type
		), pitcher_totals AS (
			SELECT
				pitcher,
				COUNT(*) AS total_rows
			FROM historical_pitches
			WHERE game_year >= %s
			  AND game_year <= %s
			  AND game_type NOT IN ('E', 'S')
			  AND pitcher IS NOT NULL
			GROUP BY pitcher
		)
		SELECT
			ptc.pitcher,
			cn.player_name,
			ptc.pitch_type,
			ptc.occurrences,
			pt.total_rows
		FROM pitch_type_counts ptc
		JOIN canonical_names cn
		  ON ptc.pitcher = cn.pitcher
		JOIN pitcher_totals pt
		  ON ptc.pitcher = pt.pitcher
		ORDER BY ptc.pitcher, ptc.occurrences DESC, ptc.pitch_type
	"""

	with get_read_cursor() as cursor:
		cursor.execute(
			query,
			(start_year, end_year, start_year, end_year, list(active_pitch_types), start_year, end_year)
		)
		rows = cursor.fetchall()

	logger.info(f"Fetched {len(rows):,} pitcher/pitch-type rows")
	return rows


def build_pitch_arsenal_json(pitcher_pitch_type_rows: list[tuple[int, str, str, int, int]]) -> Dict[str, Dict[str, Any]]:
	"""Build pitcher-indexed JSON object with pitch type occurrence counts."""
	output: Dict[str, Dict[str, Any]] = {}

	for pitcher_id, player_name, pitch_type, occurrences, total_rows in pitcher_pitch_type_rows:
		if pitch_type not in pitch_classifications:
			continue

		pitcher_key = str(pitcher_id)
		if pitcher_key not in output:
			output[pitcher_key] = {
				'player_name': player_name,
				'pitches': int(total_rows),
				'pitch_type': {},
				'pitch_type_percentage': {},
				'pitch_category': {
					'fastball': 0,
					'breaking': 0,
					'offspeed': 0,
					'other': 0,
				},
				'pitch_category_percentage': {
					'fastball': 0.0,
					'breaking': 0.0,
					'offspeed': 0.0,
					'other': 0.0,
				},
				'_total_rows': int(total_rows),
			}

		occurrences_int = int(occurrences)
		output[pitcher_key]['pitch_type'][pitch_type] = occurrences_int
		total_rows_int = output[pitcher_key]['_total_rows']
		if total_rows_int > 0:
			output[pitcher_key]['pitch_type_percentage'][pitch_type] = round(occurrences_int / total_rows_int, 6)
		else:
			output[pitcher_key]['pitch_type_percentage'][pitch_type] = 0.0

		pitch_category = pitch_classifications.get(pitch_type, 'other')
		output[pitcher_key]['pitch_category'][pitch_category] += occurrences_int

	for pitcher_data in output.values():
		total_rows_int = pitcher_data.pop('_total_rows', 0)
		for category_name, category_count in pitcher_data['pitch_category'].items():
			if total_rows_int > 0:
				pitcher_data['pitch_category_percentage'][category_name] = round(category_count / total_rows_int, 6)
			else:
				pitcher_data['pitch_category_percentage'][category_name] = 0.0

	return output


def main(start_year: int = START_YEAR, end_year: int = END_YEAR, output_dir: Path | None = None):
	"""Generate pitcher-indexed pitch arsenal JSON for the selected year range."""
	try:
		pitcher_pitch_type_rows = fetch_pitcher_pitch_type_rows(start_year, end_year)
		json_output = build_pitch_arsenal_json(pitcher_pitch_type_rows)

		if output_dir is None:
			output_dir = Path(__file__).parent
		else:
			output_dir = Path(output_dir)

		output_dir.mkdir(parents=True, exist_ok=True)
		output_filename = output_dir / f'{start_year}-{end_year}_pitch_arsenal.json'

		with open(output_filename, 'w', encoding='utf-8') as file:
			json.dump(json_output, file, indent=2, ensure_ascii=False)

		logger.info(f"✓ {output_filename} saved to disk.")
		print(f"✓ Successfully generated {output_filename}")
		print(f"  - Years: {start_year}-{end_year}")
		print(f"  - Unique pitchers: {len(json_output):,}")
		print(f"  - Output location: {output_filename.absolute()}")

	except Exception as e:
		logger.error(f"Error generating pitch arsenal JSON: {e}", exc_info=True)
		print(f"✗ Error: {e}")
		sys.exit(1)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Generate pitcher-indexed pitch arsenal JSON"
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
		help='Output directory for JSON file (default: pitch_arsenal/)'
	)

	args = parser.parse_args()

	main(
		start_year=args.start_year,
		end_year=args.end_year,
		output_dir=args.output_dir
	)
