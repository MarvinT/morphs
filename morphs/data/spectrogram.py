import morphs


def load_morph_spectrograms():
    if not morphs.paths.SPECT_PKL.exists():
        download_morph_spectrograms()
    return morphs.data.load._pickle(morphs.paths.SPECT_PKL.as_posix())


def download_morph_spectrograms():
    morphs.data.load._download(morphs.paths.SPECT_PKL, '1wirs8LQMSrc9jEaaI8P0oafMQs22Zs-l')
