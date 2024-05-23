from agentlego.types import Annotated, ImageIO, Info, VideoIO
from agentlego.utils import load_or_build_object, require

from agentlego.tools import BaseTool
from agentlego.tools.segmentation.segment_anything import load_sam_and_predictor

from mmdet.models.trackers import OCSORTTracker, ByteTracker, QuasiDenseTracker
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np 
import cv2 

class ReferringTracker(BaseTool):

    default_desc = ('The tool can track and segment objects location according to description.')
    
    @require('mmdet>=3.1.0')
    def __init__(self,
                 model: str = 'glip_atss_swin-t_b_fpn_dyhead_pretrain_obj365',
                 weight: str = '/root/justTrack/weights/glip_tiny_b_mmdet-6dfbd102.pth', 
                 sam_weight: str = '/root/justTrack/weights/sam_vit_h_4b8939.pth', 
                 device: str = 'cuda',
                 tracker: str = 'bytetrack', 
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.model = model
        self.weights = weight
        self.sam_weight = sam_weight
        self.device = device

        self.TRACKER_DICT = {
            'ocsort': OCSORTTracker, 
            'bytetrack': ByteTracker, 
            'qdtrack': QuasiDenseTracker
        }

        self.TRACKER_CONFIG_DICT = {
            'ocsort': dict(obj_score_thr=0.5, init_track_thr=0.5, ), 
            'bytetrack': dict(obj_score_thrs=dict(high=0.5, low=0.1), init_track_thr=0.5, ), 
            'qdtrack': dict(init_score_thr=0.1, obj_score_thr=0.5, ), 
        }

        self.tracker = self.TRACKER_DICT[tracker](motion=dict(type='KalmanFilter'), 
                                                  **self.TRACKER_CONFIG_DICT[tracker])

        self.top_K = 1
        self.frame_cnt = 0

        self.draw = True
        self.save_dir = './output_dir_{text_disc}'

    def setup(self):
        from mmdet.apis import DetInferencer
        self._inferencer = load_or_build_object(
            DetInferencer, model=self.model, weights=self.weights, device=self.device)
        self._visualizer = self._inferencer.visualizer

        self.sam, self.sam_predictor = load_sam_and_predictor(
            self.sam_weight, device=self.device)

    def _draw_bboxes(self, image, bboxes, ids, text, masks=None):
        """
        Draw current tracking results on image
        """
        save_dir = self.save_dir.format(text_disc=text)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
        for bbox, id in zip(bboxes, ids):
            bbox = bbox.int()
            id = id.int().item()
            x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()

            color = get_color(id)
            cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=3)
            cv2.putText(image, text=str(id), org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=color, thickness=2)

        if masks is not None:
            for idx, mask in enumerate(masks):
                id = ids[idx].int().item()
                color = np.array(get_color(id))

                mask = mask[0].cpu().numpy()
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

                image = cv2.addWeighted(image, 0.7, mask_image.astype('uint8'), 0.3, 0)
                
        cv2.imwrite(filename=os.path.join(save_dir, '{:07d}.jpg'.format(self.frame_cnt)), img=image)

    def _get_mask_with_boxes(self, image, boxes_filt):

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image.shape[:2]).to(self.device)

        features = self.sam_predictor.get_image_embedding(image)

        masks, _, _ = self.sam_predictor.predict_torch(
            features=features,
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        return masks    
    
    def _add_mask_to_image(self, ):
        pass

    def apply(
        self,
        video: VideoIO, 
        task: Annotated[str, Info("Task description, should be 'track' or 'segment'")], 
        text: Annotated[str, Info('The object description in English.')],
    ) -> Annotated[str,
                   Info('Tracked objects, include a set of bboxes in '
                        '(x1, y1, x2, y2) format, and detection scores and ids.')]:
        from mmdet.structures import DetDataSample

        need_segment = 'segment' in task.lower()

        pred_descs = []
        while not video.is_finish():
            self.frame_cnt += 1
            image_PIL = video.next_image()
            if image_PIL is None: break
            
            image = ImageIO(image_PIL)
        
            results = self._inferencer(
                image.to_array()[:, :, ::-1],
                texts=text,
                return_datasamples=True,
            )
            data_sample = results['predictions'][0]
            preds: DetDataSample = data_sample.pred_instances
            preds = preds[preds.scores > 0.3]
            preds = preds[preds.scores.topk(min(preds.scores.shape[0], self.top_K)).indices]

            data_sample = DetDataSample()
            data_sample.pred_instances = preds

            pred_track_instances = self.tracker.track(data_sample)

            bboxes = pred_track_instances.bboxes
            scores = pred_track_instances.scores
            ids = pred_track_instances.instances_id + 1
            labels = pred_track_instances.labels

            masks = None 
            if need_segment:
                masks = self._get_mask_with_boxes(image.to_array(), bboxes)

            if self.draw:
                self._draw_bboxes(image_PIL, bboxes, ids, text, masks)


            if bboxes.shape[0] == 0:
                pred_descs.append(f'frame {self.frame_cnt}, No object found.')

            else:
                pred_tmpl = '(In frame {:d}: id {:d}, bbox {:.0f}, {:.0f}, {:.0f}, {:.0f}, score {:.0f})'
                for id, bbox, score in zip(ids, bboxes, scores):
                    pred_descs.append(pred_tmpl.format(self.frame_cnt, id, bbox[0], bbox[1], bbox[2], bbox[3], score * 100))
            pred_str = '\n'.join(pred_descs)

        return pred_str

def get_color(idx):
    """
    aux func for plot_seq
    get a unique color for each id
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color