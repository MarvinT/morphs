import seaborn as sns

BEHAVE_SUBJS = ['B979', 'B1107', 'B1082', 'B1218',
                'B1222', 'B1101', 'B1088', 'B1105']
BEHAVE_COLORS = ["nice blue", "windows blue", "off blue",
                 "stormy blue", "fern", "faded green",
                 "dusty purple", "dark lilac", "red"]
BEHAVE_COLOR_MAP = {subj: color for subj, color in
                     zip(BEHAVE_SUBJS, sns.xkcd_palette(BEHAVE_COLORS))}

TRAINING = {}
for subj in ['B979', 'B1107', 'B1082', 'B1218']:
    TRAINING[subj] = 'ABCD|EFGH'
for subj in ['B1088', 'B1105']:
    TRAINING[subj] = 'ABGH|EFCD'
for subj in ['B1101', 'B1222']:
    TRAINING[subj] = 'ABEF|CDGH'

EPHYS_SUBJS = ['B1101', 'B1218', 'B1134', 'B1088',
               'st1107', 'B1096', 'B1229', 'B1082',
               'B1183']
