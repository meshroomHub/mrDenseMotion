from pathlib import Path

# Extensions of the SfMData files that AliceVision can load (sfmDataIO.load)
SFMDATA_EXTENSIONS = (".sfm", ".json", ".abc", ".usd", ".usda", ".usdc")


def isSfmDataFile(path):
    """True if path is an SfMData file readable by AliceVision (e.g. the .usda written by SfMFilter)."""
    return Path(str(path)).suffix.lower() in SFMDATA_EXTENSIONS

def get_image_paths_list(input_path):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO

    image_paths = []

    if isSfmDataFile(input_path):
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                for id, v in views.items():
                    image_paths.append(Path(v.getImage().getImagePath()))
            image_paths.sort()
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (sfmData file).")
    return image_paths
