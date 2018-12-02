import morphs


def savefig(g, name, folder=morphs.paths.FIGURES_DIR, format=None,
            formats=['png', 'pdf', 'svg', 'eps'],
            bbox_inches='tight', transparent=True, pad_inches=0):
    folder.mkdir(parents=True, exist_ok=True)
    if format:
        formats = [format]
    for format in formats:
        g.savefig((folder / (name + '.' + format)).as_posix(), format=format,
                  bbox_inches=bbox_inches, transparent=transparent, pad_inches=pad_inches)
