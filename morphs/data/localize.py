import pandas as pd
import numpy as np
import morphs
from six import exec_
from pathlib2 import Path
from joblib import Parallel, delayed


# adapted from klustakwik
# NEVER POINT THIS AT SOMETHING YOU DONT TRUST
def _read_python(path):
    assert path.exists()
    with open(path.as_posix(), "r") as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def calc_loc(block_path, squared=True):
    columns = [
        "block_path",
        "AP",
        "ML",
        "Z",
        "cluster",
        "cluster_pos",
        "cluster_accuracy",
    ]
    waveform_dict = morphs.load.waveforms()
    waveforms, cluster_map = waveform_dict[block_path]
    if waveforms is None:
        return pd.DataFrame(columns=columns)
    amps = (waveforms[:, 0, :] + waveforms[:, -1, :]) / 2 - np.min(waveforms, axis=1)
    amps /= np.max(amps, axis=0)
    if squared:
        amps = amps ** 2
    prb_files = list(Path(block_path).glob("*.prb"))
    assert len(prb_files) == 1
    prb = _read_python(prb_files[0])
    assert len(prb["channel_groups"]) == 1
    for group in prb["channel_groups"]:
        chans = prb["channel_groups"][group]["geometry"].keys()
        x, y = zip(*[prb["channel_groups"][group]["geometry"][k] for k in chans])
        y_hats = np.sum(amps * np.array(y).reshape((1, -1)), axis=1) / np.sum(
            amps, axis=1
        )

    d = {}
    d["block_path"] = block_path
    d["AP"], d["ML"], d["Z"] = morphs.data.parse.recording_site(block_path)
    i_cluster_map = {v: k for k, v in cluster_map.items()}
    d["cluster"] = [i_cluster_map[i] for i in range(len(cluster_map))]
    d["cluster_pos"] = y_hats
    _, cluster_accuracies = morphs.load.cluster_accuracies()
    d["cluster_accuracy"] = (
        cluster_accuracies[block_path].loc[d["cluster"]]["accuracy"].values
    )
    return pd.DataFrame(data=d, columns=columns)


def generate_all_loc(parallel=False, n_jobs=morphs.parallel.N_JOBS):
    if parallel and n_jobs > 1:
        all_locs = Parallel(n_jobs=n_jobs)(
            delayed(calc_loc)(block_path) for block_path in morphs.paths.blocks()
        )
    else:
        all_locs = [calc_loc(block_path) for block_path in morphs.paths.blocks()]
    all_locs_df = pd.concat(all_locs, ignore_index=True)
    all_locs_df["block_path"] = all_locs_df["block_path"].astype("category")
    all_locs_df.to_pickle(morphs.paths.LOCATIONS_PKL)


@morphs.utils.load._load(
    morphs.paths.LOCATIONS_PKL,
    generate_all_loc,
    download_func=morphs.utils.load._download(
        morphs.paths.LOCATIONS_PKL, "1wLoMiKJjKPQbNF_qplqrMzHLyFCyFXn3"
    ),
)
def load_all_loc(prefer_download=True):
    return pd.read_pickle(morphs.paths.LOCATIONS_PKL.as_posix())
