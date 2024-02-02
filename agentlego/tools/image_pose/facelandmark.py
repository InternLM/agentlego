from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class HumanFaceLandmark(BaseTool):
    """A tool to extract human face landmarks from an image.

    Args:
        model (str): The model name used to inference. Which can be found
            in the ``MMPose`` repository. Defaults to 'face'.
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can estimate the landmark or keypoints of '
                    'human faces in an image and draw the landmarks image.')

    @require('mmpose')
    def __init__(self, model: str = 'face', device: str = 'cuda', toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model_name = model
        self.device = device

    def setup(self):
        from mmpose.apis import MMPoseInferencer
        self._inferencer = load_or_build_object(
            MMPoseInferencer, pose2d=self.model_name, device=self.device)

    def apply(self, image: ImageIO
              ) -> Annotated[ImageIO, Info('The human face landmarks image.')]:
        image = image.to_array()[:, :, ::-1]
        results = next(
            self._inferencer(
                inputs=image,
                skeleton_style='mmpose',
                black_background=True,
                return_vis=True,
            ))
        landmarks = results['visualization'][0][:, :, ::-1]
        return ImageIO(landmarks)
