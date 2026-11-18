__version__ = "2.0"

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from pyalicevision import parallelization as avpar

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

    size = avpar.DynamicViewsSize("inputImages")
    parallelization = DenseMotionBlockSize()

    inputs = [
        desc.File(
            name="inputImages",
            label="Input Images",
            description="sfmData filepath",
            value="",
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
        )
    ]

    def processChunk(self, chunk):
        import cv2 as cv
        from img_proc import image
        import os
        import numpy as np
        from pathlib import Path
        from img_proc.sfmData import get_image_paths_list
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.inputImages.value:
                chunk.logger.warning('No input folder given.')

            frameStop = chunk.range.end if chunk.range.iteration == chunk.range.fullSize-1 else chunk.range.end + 1

            chunk_image_paths_full = get_image_paths_list(chunk.node.inputImages.value)
            if not chunk_image_paths_full:
                raise FileNotFoundError(f'No image files found in {chunk.node.inputImages.value}')
            chunk_image_paths = chunk_image_paths_full[chunk.range.start:frameStop]

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
