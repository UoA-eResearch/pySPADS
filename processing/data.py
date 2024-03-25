from pathlib import Path

import pandas as pd

from pipeline.decompose import parse_filename
from processing.recomposition import epoch_index_to_datetime


def load_imfs(folder: Path) -> dict[tuple[str, float], pd.DataFrame]:
    """
    Load IMFs from a folder, parse label and noise from filenames
    :param folder: folder containing IMF files
    :return: dict of IMFs, with keys (label, noise)
    """
    assert folder.is_dir(), f'Folder {folder} does not exist'
    imfs = {}
    for file in folder.glob('*.csv'):
        label, noise = parse_filename(file)
        imfs[(label, noise)] = pd.read_csv(file, index_col=0, parse_dates=True)

        # TODO: temporary until data is regenerated
        for key in imfs:
            if imfs[key].index.inferred_type == 'integer':
                imfs[key] = epoch_index_to_datetime(imfs[key])

        # Convert column names to ints
        imfs[(label, noise)].columns = imfs[(label, noise)].columns.astype(int)
    return imfs
