__version__ = "1.0"

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class DenseMotionNodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        from pathlib import Path
        import itertools

        input_path_param = node.attribute(self._params[0])
        extension_param = node.attribute(self._params[1])

        input_path = input_path_param.value
        extension = extension_param.value
        include_suffixes = [extension.lower(), extension.upper()]

        size = 1
        if Path(input_path).is_dir():
            image_paths = list(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
            size = len(image_paths)
        elif node.attribute(self._params[0]).isLink:
            size = node.attribute(self._params[0]).getLinkParam().node.size
        
        return size


class DenseMotionBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class DenseMotion(desc.Node):
    category = "Dense Motion"
    documentation = """This node computes an optical flow from a monocular image sequence.
                    The openCV implementation of the Farneback algorithm is used."""
    
    gpu = desc.Level.INTENSIVE

    size = DenseMotionNodeSize(['inputImages', 'inputExtension'])
    parallelization = DenseMotionBlockSize()

    inputs = [
        desc.File(
            name="inputImages",
            label="Input Images",
            description="Input images to estimate the depth from. Folder path or sfmData filepath",
            value="",
        ),
        desc.ChoiceParam(
            name="inputExtension",
            label="Input Extension",
            description="Extension of the input images. This will be used to determine which images are to be used if \n"
                        "a directory is provided as the input. If \"\" is selected, the provided input will be used as such.",
            values=["jpg", "jpeg", "png", "exr"],
            value="exr",
            exclusive=True,
        ),
        desc.FloatParam(
            name="PyramidScale",
            label="Pyramid Scale",
            value=0.5,
            description="Image scale to build pyramids for each image.",
            range=(0.1, 0.9, 0.1),
        ),
        desc.IntParam(
            name="Levels",
            label="Levels",
            value=3,
            description="Number of pyramid layers.",
            range=(1, 10, 1),
        ),
        desc.IntParam(
            name="WindowSize",
            label="Window Size",
            value=15,
            description="Averaging window size.",
            range=(3, 30, 1),
        ),
        desc.IntParam(
            name="IterationNumber",
            label="Iteration Number",
            value=3,
            description="Number of iterations at each pyramid level.",
            range=(1, 10, 1),
        ),
        desc.IntParam(
            name="blockSize",
            label="Block Size",
            value=50,
            description="Sets the number of images to process in one chunk. If set to 0, all images are processed at once.",
            range=(0, 1000, 1),
            enabled=lambda node: node.model.value == "MoGe"
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name='output',
            label='Output Folder',
            description="Output folder containing the flow saved as exr images.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="OpticalFlow",
            label="Optical Flow",
            description="Output optical flow images",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/<FILESTEM>.exr",
            group="",
        )
    ]

    def preprocess(self, node):
        extension = node.inputExtension.value
        input_path = node.inputImages.value

        image_paths = get_image_paths_list(input_path, extension)

        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')

        self.image_paths = image_paths

    def processChunk(self, chunk):
        import cv2 as cv
        from img_proc import image
        import os
        import numpy as np
        from pathlib import Path
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.inputImages.value:
                chunk.logger.warning('No input folder given.')

            frameStop = chunk.range.end if chunk.range.iteration == chunk.range.fullSize-1 else chunk.range.end + 1

            chunk_image_paths = self.image_paths[chunk.range.start:frameStop]

            # computation
            chunk.logger.info(f'Starting computation on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)}...')

            for idx, path in enumerate(chunk_image_paths):
                frame1, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx]), True)
                if idx == 0:
                    prvs = cv.cvtColor(255.0*frame1, cv.COLOR_BGR2GRAY)
                else:
                    next = cv.cvtColor(255.0*frame1, cv.COLOR_BGR2GRAY)
                    flow = cv.calcOpticalFlowFarneback(prvs, next, None,
                                                       chunk.node.PyramidScale.value,
                                                       chunk.node.Levels.value,
                                                       chunk.node.WindowSize.value,
                                                       chunk.node.IterationNumber.value,
                                                       5, 1.2, 0)

                    h,w,_ = flow.shape
                    nc = np.zeros((h,w,1), dtype=flow.dtype)
                    flow = np.concatenate((flow,nc), axis=2)

                    outputDirPath = Path(chunk.node.output.value)
                    image_stem = Path(chunk_image_paths[idx]).stem
                    of_file_name = image_stem + ".exr"

                    image.writeImage(str(outputDirPath / of_file_name), flow, h_ori, w_ori, orientation, pixelAspectRatio)

                    prvs = next
            
            chunk.logger.info('Publish end')
        finally:
            chunk.logManager.end()

def get_image_paths_list(input_path, extension):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO
    from pathlib import Path
    import itertools

    include_suffixes = [extension.lower(), extension.upper()]
    image_paths = []

    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
    elif input_path[-4:].lower() == ".sfm" or input_path[-4:].lower() == ".abc":
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                for id, v in views.items():
                    image_paths.append(Path(v.getImage().getImagePath()))
            image_paths.sort()
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (folder or sfmData file).")
    return image_paths
