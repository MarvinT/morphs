import morphs
import pytest


@pytest.mark.run(order=4)
def test_download_behavior_df():
    morphs.data.behavior.download_behavior_df()
    assert morphs.paths.BEHAVE_PKL.exists()


@pytest.mark.run(order=5)
def test_behav_data_inverted():
    behavior_df = morphs.data.load.behavior_df()
    morphs.data.parse.stim_id(behavior_df)
    behavior_df = morphs.data.parse.behav_data_inverted(behavior_df)

    for (subj, morph_dim), group in behavior_df.groupby(['subj', 'morph_dim'], observed=True):
        behavior_df.loc[group.index, 'inverted_slow'] = morphs.data.parse.is_inverted_dim(
            subj, morph_dim)

    assert np.all(behavior_df['inverted'] == behavior_df['inverted_slow'])


@pytest.mark.run(order=6)
def test_generate_psychometric_params():
    morphs.data.psychometric.generate_psychometric_params()
    assert morphs.paths.PSYCHOMETRIC_PKL.exists()
