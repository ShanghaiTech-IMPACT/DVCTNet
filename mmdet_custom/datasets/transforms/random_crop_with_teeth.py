from typing import Tuple,Union
from mmdet.datasets.transforms import RandomCrop
import numpy as np
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class RandomCropWithTeeth(RandomCrop):
    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape[:2]
        
        bboxes = np.array([t["region_bbox"] for t in results["teeth"]], dtype=np.float32)
        img_h, img_w = img_shape[:2]

        # 平移
        bboxes[:, [0, 2]] -= offset_w
        bboxes[:, [1, 3]] -= offset_h

        # clip
        if self.bbox_clip_border:
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img_w)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img_h)

        # mask: 部分在图像内的才保留
        mask = (bboxes[:, 0] < img_w) & (bboxes[:, 1] < img_h) & \
            (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)

        # 构建结果
        new_teeth = [
            {"region_bbox": box.tolist(), "image_path": t["image_path"]}
            for box, t in zip(bboxes, results["teeth"]) if mask.any()
        ]

        results["teeth"] = new_teeth
        # for tooth in results["teeth"]:
        #     x1,y1,x2,y2 = tooth['region_bbox']
        #     x1,x2 = x1 - offset_w, x2 - offset_w
        #     y1,y2 = y1 - offset_h, y2 - offset_h
        #     img_h,img_w = img_shape[:2]
        #     if self.bbox_clip_border:
        #         x1 = np.clip(x1,0,img_w)
        #         x2 = np.clip(x2,0,img_w)
        #         y1 = np.clip(y1,0,img_h)
        #         y2 = np.clip(y2,0,img_h)
        #     if x1<img_w and y1<img_h and x2>0 and y2>0:
        #         new_teeth.append({"region_bbox": [x1,y1,x2,y2], 
        #                           "image_path": tooth["image_path"]})
                
            
        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

            # We should remove the instance ids corresponding to invalid boxes.
            if results.get('gt_instances_ids', None) is not None:
                results['gt_instances_ids'] = \
                    results['gt_instances_ids'][valid_inds]

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                          crop_x1:crop_x2]

        return results