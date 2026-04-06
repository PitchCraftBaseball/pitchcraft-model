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


def _empty_pitch_stats() -> Dict[str, Any]:
	return {
		'pitches': 0,
		'pitch_type': {},
		'pitch_type_percentage': {},
		'pitch_group': {
			'fastball': 0,
			'breaking': 0,
			'offspeed': 0,
			'other': 0,
		},
		'pitch_group_percentage': {
			'fastball': 0.0,
			'breaking': 0.0,
			'offspeed': 0.0,
			'other': 0.0,
		},
	}


def fetch_pitcher_pitch_type_rows(start_year: int = START_YEAR, end_year: int = END_YEAR) -> list[tuple[int, str, int, str, int, int]]:
	"""
	Fetch pitcher rows with canonical player name, year, pitch type occurrence counts,
	and total rows for each pitcher/year.

	Each returned row is:
	(pitcher_id, canonical_player_name, game_year, pitch_type, occurrences, year_total_rows)
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
				game_year,
				pitch_type,
				COUNT(*) AS occurrences
			FROM historical_pitches
			WHERE game_year >= %s
			  AND game_year <= %s
			  AND game_type NOT IN ('E', 'S')
			  AND pitcher IS NOT NULL
			  AND pitch_type IS NOT NULL
			  AND pitch_type = ANY(%s)
			GROUP BY pitcher, game_year, pitch_type
		), pitcher_year_totals AS (
			SELECT
				pitcher,
				game_year,
				COUNT(*) AS total_rows
			FROM historical_pitches
			WHERE game_year >= %s
			  AND game_year <= %s
			  AND game_type NOT IN ('E', 'S')
			  AND pitcher IS NOT NULL
			GROUP BY pitcher, game_year
		)
		SELECT
			ptc.pitcher,
			cn.player_name,
			ptc.game_year,
			ptc.pitch_type,
			ptc.occurrences,
			pt.total_rows
		FROM pitch_type_counts ptc
		JOIN canonical_names cn
		  ON ptc.pitcher = cn.pitcher
		JOIN pitcher_year_totals pt
		  ON ptc.pitcher = pt.pitcher
		 AND ptc.game_year = pt.game_year
		ORDER BY ptc.pitcher, ptc.game_year, ptc.occurrences DESC, ptc.pitch_type
	"""

	with get_read_cursor() as cursor:
		cursor.execute(
			query,
			(start_year, end_year, start_year, end_year, list(active_pitch_types), start_year, end_year)
		)
		rows = cursor.fetchall()

	logger.info(f"Fetched {len(rows):,} pitcher/year/pitch-type rows")
	return rows


def build_pitch_arsenal_json(pitcher_pitch_type_rows: list[tuple[int, str, int, str, int, int]]) -> Dict[str, Dict[str, Any]]:
	"""Build pitcher-indexed JSON object with year-separated pitch stats."""
	output: Dict[str, Dict[str, Any]] = {}

	for pitcher_id, player_name, game_year, pitch_type, occurrences, total_rows in pitcher_pitch_type_rows:
		if pitch_type not in pitch_classifications:
			continue

		pitcher_key = str(pitcher_id)
		if pitcher_key not in output:
			output[pitcher_key] = {
				'player_name': player_name,
			}

		year_key = str(game_year)
		if year_key not in output[pitcher_key]:
			output[pitcher_key][year_key] = _empty_pitch_stats()
			output[pitcher_key][year_key]['pitches'] = int(total_rows)

		occurrences_int = int(occurrences)
		year_stats = output[pitcher_key][year_key]
		year_stats['pitch_type'][pitch_type] = occurrences_int
		total_rows_int = year_stats['pitches']
		if total_rows_int > 0:
			year_stats['pitch_type_percentage'][pitch_type] = round(occurrences_int / total_rows_int, 6)
		else:
			year_stats['pitch_type_percentage'][pitch_type] = 0.0

		pitch_group = pitch_classifications.get(pitch_type, 'other')
		year_stats['pitch_group'][pitch_group] += occurrences_int

	for pitcher_data in output.values():
		for key, year_stats in list(pitcher_data.items()):
			if key == 'player_name':
				continue
			for category_name, category_count in year_stats['pitch_group'].items():
				if year_stats['pitches'] > 0:
					year_stats['pitch_group_percentage'][category_name] = round(category_count / year_stats['pitches'], 6)
				else:
					year_stats['pitch_group_percentage'][category_name] = 0.0

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
