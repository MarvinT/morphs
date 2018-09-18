'''Parsing functions for the stimuli names for this project'''
import morphs


def stim_id(df, stim_id='stim_id', end='end', morph_dim='morph_dim',
            morph_pos='morph_pos', lesser_dim='lesser_dim',
            greater_dim='greater_dim'):
    df[end] = df[stim_id].isin(list('abcdefghi'))
    df[morph_dim] = df[~df[end]][stim_id].str[0:2]
    df[morph_pos] = df[~df[end]][stim_id].str[2:].astype(int)
    df[lesser_dim] = df[~df[end]][morph_dim].str[0]
    df[greater_dim] = df[~df[end]][morph_dim].str[1]


def separate_endpoints(stim_id_series):
    stim_ids = stim_id_series.str.replace('[a-i]001', '')
    for motif in 'abcdefgh':
        stim_ids = stim_ids.str.replace('[a-i]%s128' % (motif), motif)
    end = stim_ids.isin(list('abcdefghi'))
    return stim_ids, end


def is_inverted_dim(subj, morph_dim):
    left, right = morphs.subj.TRAINING[subj].lower().split('|')
    les, gre = morph_dim
    assert (les in left) != (gre in left)
    assert (les in right) != (gre in right)
    return gre in left


def behav_data_stim_id(df, subj):
    df = df[(df['response'] != 'none') & (df['type_'] == 'normal')]
    df['stim_id'] = df['stimulus'].str.split('/').str[-1].str[:-4]
    df = df[df['stim_id'].str.len() == 5]
    df = df[df['stim_id'].str[1:] != '_rec']
    df['subj'] = subj
    return df


def behav_data_inverted(df):
    # apparently groupby with categorical dtype is broken
    df['class_'] = df['class_'].astype(str)
    inverted_map = df[(df['morph_pos'] == 1)].groupby(['subj', 'morph_dim'],
                                                      observed=True).agg(lambda x: x.iloc[0])['class_'] == 'R'
    df = df.join(inverted_map.to_frame(name='inverted'), on=(
        'subj', 'morph_dim'), how='left', sort=False)
    df['greater_response'] = (df['response'] == 'R') != (df['inverted'])
    return df
