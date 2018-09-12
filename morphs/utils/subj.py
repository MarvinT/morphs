import seaborn as sns

behave_subjs = ['B979', 'B1107', 'B1082', 'B1218', 'B1222', 'B1101', 'B1088', 'B1105']
behave_colors = ["nice blue", "windows blue", "off blue", "stormy blue", "fern", "faded green", "dusty purple", "dark lilac", "red"]
behave_color_map =  {subj:color for subj, color in zip(behave_subjs, sns.xkcd_palette(behave_colors))}

training = {}
for subj in ['B979', 'B1107', 'B1082', 'B1218']:
    training[subj] = 'ABCD|EFGH'
for subj in ['B1088', 'B1105']:
    training[subj] = 'ABGH|EFCD'
for subj in ['B1101', 'B1222']:
    training[subj] = 'ABEF|CDGH'

ephys_subjs = [
          'B1101',
          'B1218', 
          'B1134',
          # 'B1055', 
          'B1088',
          'st1107',
          'B1096',
          'B1229',
          'B1082',
          'B1183'
               ]