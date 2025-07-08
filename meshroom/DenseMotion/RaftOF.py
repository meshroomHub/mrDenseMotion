__version__ = "1.0"

from re import M
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class RaftOFNodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        input_path_param = node.attribute(self._params[0])
        extension_param = node.attribute(self._params[1])

        extension = extension_param.value
        input_path = input_path_param.value
        image_paths = get_image_paths_list(input_path, extension)

        return(max(1, len(image_paths)))


class RaftOFBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class RaftOF(desc.Node):
    category = "Dense Motion"
    documentation = """This node computes an optical flow from a monocular image sequence.
                    The deep model RAFT is used."""
    
    gpu = desc.Level.INTENSIVE

    size = RaftOFNodeSize(['inputImages', 'inputExtension'])
    parallelization = RaftOFBlockSize()

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
        desc.File(
            name="modelPath",
            label="Model Path",
            description="Weights file for the deep model.",
            value="${RAFT_OPTICALFLOW_MODEL_PATH}",
        ),
        desc.IntParam(
            name="maxImageSize",
            label="Max Image Size",
            value=1024,
            description="Maximum size for the largest image dimension.",
            range=(256, 8192, 8),
        ),
        desc.IntParam(
            name="IterationNumber",
            label="Iteration Number",
            value=20,
            description="Number of iterations.",
            range=(1, 50, 1),
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
            semantic="imageList",
            value="{nodeCacheFolder}/*.exr",
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
        from raft import RAFT
        from utils.utils import InputPadder
        import argparse
        import torch
        from img_proc import image
        import os
        import numpy as np
        from pathlib import Path
        import OpenImageIO as oiio
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.inputImages.value:
                chunk.logger.warning('No input folder given.')

            frameStop = chunk.range.end if chunk.range.iteration == chunk.range.fullSize-1 else chunk.range.end + 1

            chunk_image_paths = self.image_paths[chunk.range.start:frameStop]

            parser = argparse.ArgumentParser()
            parser.add_argument('--model', help="restore checkpoint")
            parser.add_argument('--small', action='store_true', help='use small model')
            parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
            parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
            args = parser.parse_args(['--model=' + chunk.node.modelPath.evalValue])

            model = torch.nn.DataParallel(RAFT(args))
            model.load_state_dict(torch.load(args.model))

            model = model.module
            DEVICE = 'cuda'
            model.to(DEVICE)
            model.eval()

            # computation
            chunk.logger.info(f'Starting computation on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)}...')

            for idx, path in enumerate(chunk_image_paths):
                if idx > 0:
                    with torch.no_grad():
                        image1, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx - 1]), True)
                        image2, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx]), True)

                        if max(h_ori, w_ori) > chunk.node.maxImageSize.value:
                            maxDim = float(max(h_ori, w_ori))
                            scale = float(chunk.node.maxImageSize.value) / maxDim
                            h_tgt = int(h_ori * scale)
                            w_tgt = int(w_ori * scale)
                            chnb = image1.shape[2]

                            oiio_image1_buf = oiio.ImageBuf(image1)
                            oiio_image1_buf = oiio.ImageBufAlgo.resize(oiio_image1_buf, roi=oiio.ROI(0, w_tgt, 0, h_tgt, 0, 1, 0, chnb+1))
                            image1 = oiio_image1_buf.get_pixels(format=oiio.FLOAT)
                            oiio_image2_buf = oiio.ImageBuf(image2)
                            oiio_image2_buf = oiio.ImageBufAlgo.resize(oiio_image2_buf, roi=oiio.ROI(0, w_tgt, 0, h_tgt, 0, 1, 0, chnb+1))
                            image2 = oiio_image2_buf.get_pixels(format=oiio.FLOAT)

                        image1 = torch.from_numpy(255.0*image1).permute(2,0,1).float().unsqueeze(0)
                        image2 = torch.from_numpy(255.0*image2).permute(2,0,1).float().unsqueeze(0)
                        image1 = image1.to(DEVICE)
                        image2 = image2.to(DEVICE)

                        padder = InputPadder(image2.shape)
                        image1, image2 = padder.pad(image1, image2)

                        flow_low, flow_up = model(image1, image2, iters=chunk.node.IterationNumber.value, test_mode=True)
                        flow_up = padder.unpad(flow_up)
                        flow_up = flow_up[0].permute(1,2,0).cpu().numpy()
                        flow_out = flow_up.copy()

                        h,w,_ = flow_out.shape
                        nc = np.zeros((h,w,1), dtype=flow_out.dtype)
                        flow_out = np.concatenate((flow_out,nc), axis=2)

                        outputDirPath = Path(chunk.node.output.value)
                        image_stem = Path(chunk_image_paths[idx]).stem
                        of_file_name = image_stem + ".exr"

                        image.writeImage(str(outputDirPath / of_file_name), flow_out, h_ori, w_ori, orientation, pixelAspectRatio)
            
            chunk.logger.info('Publish end')
        finally:
            chunk.logManager.end()

def get_image_paths_list(input_path, extension):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO
    from pathlib import Path
    import itertools

    include_suffices = [extension.lower(), extension.upper()]
    image_paths = []

    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffices)))
    elif input_path[-4:].lower() == ".sfm":
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
