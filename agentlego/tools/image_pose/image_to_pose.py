from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class HumanBodyPose(BaseTool):
    """A tool to extract human body keypoints from an image.

    Args:
        model (str): The model name used to inference. Which can be found
            in the ``MMPose`` repository.
            Defaults to `human`.
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can estimate the pose or keypoints of '
                    'human in an image and draw the human pose image.')

    @require('mmpose')
    def __init__(self, model: str = 'human', device: str = 'cuda', toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model_name = model
        self.device = device

    def setup(self):
        from mmpose.apis import MMPoseInferencer
        self._inferencer = load_or_build_object(
            MMPoseInferencer, pose2d=self.model_name, device=self.device)

    def apply(self, image: ImageIO
              ) -> Annotated[ImageIO, Info('The human pose keypoints image.')]:
        image = image.to_array()[:, :, ::-1]
        vis_params = self.adaptive_vis_params(*image.shape[:2])
        results = next(
            self._inferencer(
                inputs=image,
                skeleton_style='openpose',
                black_background=True,
                return_vis=True,
                **vis_params,
            ))
        skeleton_image = results['visualization'][0][:, :, ::-1]
        return ImageIO(skeleton_image)

    @staticmethod
    def adaptive_vis_params(width, height) -> dict:
        scale = (width * height)**0.5

        radius = max(round((3 / 256) * scale), 3)
        thickness = max(round((1 / 256) * scale), 3)

        return dict(radius=int(radius), thickness=int(thickness))
