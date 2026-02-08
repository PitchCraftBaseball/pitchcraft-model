import pybaseball
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import json


pybaseball.cache.enable()

latest_column_added = 'description_cat'

years = ['2021', '2022', '2023', '2024', '2025']

start_dates = {'2021': '2021-04-01',
               '2022': '2022-04-07',
               '2023': '2023-03-30',
               '2024': '2024-03-20',
               '2025': '2025-03-18'}
end_dates = {'2021': '2021-11-02',
             '2022': '2022-11-05',
             '2023': '2023-11-01',
             '2024': '2024-10-30',
             '2025': '2025-11-01'}

categorize_description = {'hit_into_play': 'P',
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
                          'bunt_foul_tip': 'S',}

categorize_description = defaultdict(lambda: 'X', categorize_description)

# Possible outs, counts, and base situations
p_outs = [0,1,2]
p_counts = ['00', '01', '02', '10', '11', '12', '20', '21', '22', '30', '31', '32']
p_bases = ['XXX', 'OXX', 'XOX', 'OOX', 'XXO', 'OXO', 'XOO', 'OOO']

def generate_count(x):
    """uses balls and strikes"""
    return str(x[0]) + str(x[1])

def generate_inning_code(x):
    """uses game_pk, inning, inning_topbot"""
    return str(x[0]) + str(x[1]) + str(x[2])

def situation_to_identifier(x):
    """uses outs, counts, bases"""
    first = x[2]
    second = x[3]
    third = x[4]
    output = str(x[0]) + x[1]
    for c in [first, second, third]:
        if c:
            output += 'O'
        else:
            output += 'X'
    return output


def pitch_logic(key, result):
    """Takes a situation identifier and determines the new situation based on whether the outcome is S, B, or F"""
    outs = int(key[0])
    balls = int(key[1])
    strikes = int(key[2])
    first = key[3] == 'O'
    second = key[4] == 'O'
    third = key[5] == 'O'
    extra = ''
    if result == 'B':
        balls += 1
        if balls >= 4:
            balls = 0
            strikes = 0
            if first:
                if second:
                    if third:
                        extra = '+' 
                    else:
                        third = True
                else:
                    second = True
            else:
                first = True
    elif result == 'S':
        strikes += 1
        if strikes >= 3:
            balls = 0
            strikes = 0
            outs += 1
            if outs == 3:
                return 'INNING_OVER'
    elif result == 'F':
        if strikes == 2:
            return key
        else:
            strikes += 1
                        
        
    output = f'{outs}{balls}{strikes}'
    for runner in [first, second, third]:
        if runner:
            output += 'O'
        else:
            output += 'X'
    return output + extra
    

all_dfs = []

for year in years:
    print('beginning process for year ' + year)
    filename = year + '_mlb_statcast.csv'
    if os.path.isfile(filename):
        print('local data found.')
        df = pd.read_csv(filename)
        print('dataframe loaded')
    else:
        print('local data not found. downloading statcast data...')
        df = pybaseball.statcast(start_dt = start_dates[year], end_dt = end_dates[year]).reset_index(drop = True)
        df.to_csv(filename)
        print('local data saved.')

    if latest_column_added not in df.columns:
        # df preprocessing. will cache when done.
        df['count'] = df[['balls', 'strikes']].apply(generate_count, axis = 1)

        print('generating inning codes. this may take a while...')
        df['inning_code'] = df[['game_pk', 'inning', 'inning_topbot']].apply(generate_inning_code, axis = 1)
        inning_codes = df['inning_code'].unique()
        print('inning codes generated.')

        runs_to_score = {}
        i = 0
        j = len(inning_codes)
        for inning_code in inning_codes:
            print(f'determining runs scored in each inning -- iteration {i}/{j}', end = '\r')
            dfinn = df.loc[df['inning_code'] == inning_code]
            champ = dfinn.sort_values(by = 'at_bat_number', ascending = False).iloc[0]['post_bat_score']
            runs_to_score[inning_code] = champ
            i += 1
        print('runs scored in each inning successfully determined.')

        df['post_inn_score'] = df['inning_code'].apply(lambda x: runs_to_score[x])
        df['runs_to_score'] = df['post_inn_score'] - df['bat_score']
        print('remaining runs to score per situation calculated.')

        for c in ['on_1b', 'on_2b', 'on_3b']:
            df[c] = df[c].fillna(0)
        df['rofirst'] = df['on_1b'].apply(lambda x: x > 0)
        df['rosecond'] = df['on_2b'].apply(lambda x: x > 0)
        df['rothird'] = df['on_3b'].apply(lambda x: x > 0)
        df['situation_identifier'] = df[['outs_when_up', 'count', 'rofirst', 'rosecond', 'rothird']].apply(situation_to_identifier, axis = 1)
        df['description_cat'] = df['description'].apply(lambda x: categorize_description[x])
        print('pitch descriptions categorized.')

        print('preprocessing complete.')
        df.to_csv(filename)
        print('local data saved.')
    else:
        print("column '" + latest_column_added + "' found, skipping preprocessing steps.")

    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)
pitch_types = sorted(df_all['pitch_type'].dropna().unique().tolist())

json_output = {}

for outs in p_outs:
    json_output[outs] = {}
    for bases in p_bases:
        json_output[outs][bases] = {}
        for count in p_counts:
            count_format = count[0] + '-' + count[1]
            key = str(outs) + count + bases

            view = df_all.loc[df_all['situation_identifier'] == key]
            pitch_entries = []
            for pitch_type in pitch_types:
                pitch_view = view.loc[view['pitch_type'] == pitch_type]
                occurrences = len(pitch_view)
                if occurrences > 0:
                    run_expectancy = round(pitch_view['runs_to_score'].mean(), 3)
                    pitch_entries.append((pitch_type, int(occurrences), run_expectancy))

            pitch_entries.sort(key=lambda x: x[1], reverse=True)
            json_output[outs][bases][count_format] = {}
            for pitch_type, occurrences, run_expectancy in pitch_entries:
                json_output[outs][bases][count_format][pitch_type] = {
                    'occurrences': occurrences,
                    'run_expectancy': run_expectancy
                }

output_filename = f'{min(years)}-{max(years)}_re288_pitch.json'
with open(output_filename, 'w') as file:
    json.dump(json_output, file, indent=2)

print(f'{output_filename} saved to disk.')
