import morphs


@morphs.utils.load._load(morphs.paths.SPECT_PKL, None,
                         download_func=morphs.utils.load._download(morphs.paths.SPECT_PKL,
                                                                   '1wirs8LQMSrc9jEaaI8P0oafMQs22Zs-l'))
def load_morph_spectrograms():
    return morphs.utils.load._pickle(morphs.paths.SPECT_PKL.as_posix())
