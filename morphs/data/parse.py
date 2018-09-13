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
