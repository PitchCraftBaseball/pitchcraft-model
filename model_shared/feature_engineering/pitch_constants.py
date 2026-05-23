SWING_CODE = ['foul_bunt', 'foul', 'hit_into_play', 'swinging_strike', 'foul_tip',
                'swinging_strike_blocked', 'missed_bunt', 'bunt_foul_tip']
WHIFF_CODE = ['swinging_strike', 'foul_tip', 'swinging_strike_blocked']

FASTBALL = {'FF', 'SI', 'FC'}
BREAKING = {'SL', 'CU', 'KC', 'SV', 'ST'}
OFFSPEED = {'CH', 'FS'}

IGNORE = {"ABS", "PO", "FA", 'EP'}

PAD_ID = 0

MAX_LEN = 8

BASE_LABELS = {
    'XXX': 0,
    'OXX': 1,
    'XOX': 2,
    'OOX': 3,
    'XXO': 4,
    'OXO': 5,
    'XOO': 6,
    'OOO': 7,
}
