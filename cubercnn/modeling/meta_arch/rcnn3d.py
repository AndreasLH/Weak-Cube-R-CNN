# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import Dict, List, Optional
from detectron2.layers import move_device_like
from detectron2.structures.image_list import ImageList
import torch
import numpy as np
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.data import MetadataCatalog

from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.utils.logger import _log_api_usage
from detectron2.modeling.meta_arch import (
    META_ARCH_REGISTRY, GeneralizedRCNN
)
from cubercnn.data.generate_depth_maps import setup_depth_model
from cubercnn.modeling.roi_heads import build_roi_heads

from detectron2.data import MetadataCatalog
from pytorch3d.transforms import rotation_6d_to_matrix
from cubercnn.modeling.roi_heads import build_roi_heads
from cubercnn import util, vis
import torch.nn.functional as F
from detectron2.config import configurable
import torch.nn as nn

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class RCNN3D(GeneralizedRCNN):
    
    @classmethod
    def from_config(cls, cfg, priors=None):
        backbone = build_backbone(cfg, priors=priors)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape(), priors=priors),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        # scaling factor for the sample relative to its original scale
        # e.g., how much has the image been upsampled by? or downsampled?
        im_scales_ratio = [info['height'] / im.shape[1] for (info, im) in zip(batched_inputs, images)]

        # The unmodified intrinsics for the image
        Ks = [torch.FloatTensor(info['K']) for info in batched_inputs]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # the backbone is actually a FPN, where the DLA model is the bottom-up structure.
        # FPN: https://arxiv.org/abs/1612.03144v2
        # backbone and proposal generator only work on 2D images and annotations.
        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)

        instances, detector_losses = self.roi_heads(
            images, features, proposals, 
            Ks, im_scales_ratio, 
            gt_instances
        )

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0 and storage.iter > 0:
                self.visualize_training(batched_inputs, proposals, instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)

        # scaling factor for the sample relative to its original scale
        # e.g., how much has the image been upsampled by? or downsampled?
        im_scales_ratio = [info['height'] / im.shape[1] for (info, im) in zip(batched_inputs, images)]
        
        # The unmodified intrinsics for the image
        Ks = [torch.FloatTensor(info['K']) for info in batched_inputs]

        features = self.backbone(images.tensor)

        # Pass oracle 2D boxes into the RoI heads
        if type(batched_inputs == list) and np.any(['oracle2D' in b for b in batched_inputs]):
            oracles = [b['oracle2D'] for b in batched_inputs]
            results, _ = self.roi_heads(images, features, oracles, Ks, im_scales_ratio, None)
        
        # normal inference
        else:
            proposals, _ = self.proposal_generator(images, features, None)
            results, _ = self.roi_heads(images, features, proposals, Ks, im_scales_ratio, None)
            
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def visualize_training(self, batched_inputs, proposals, instances):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
            instances (list): a list that contains predicted RoIhead instances. Both
                batched_inputs and proposals should have the same length.
        """
        
        storage = get_event_storage()

        # minimum number of boxes to try to visualize per image
        max_vis_prop = 20

        if not hasattr(self, 'thing_classes'):
            self.thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
            self.num_classes = len(self.thing_classes)

        for input, prop, instances_i in zip(batched_inputs, proposals, instances):

            img = input["image"]            
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            img_3DGT = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]]) # BGR
            img_3DPR = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]]) # BGR

            '''
            Visualize the 2D GT and proposal predictions
            '''
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img_rpn = np.concatenate((anno_img, prop_img), axis=1)
            vis_img_rpn = vis_img_rpn.transpose(2, 0, 1)
            storage.put_image("Left: GT 2D bounding boxes; Right: Predicted 2D proposals", vis_img_rpn)

            '''
            Visualize the 3D GT and predictions
            '''
            K = torch.tensor(input['K'], device=self.device)
            scale = input['height']/img.shape[0]
            fx, sx = (val.item()/scale for val in K[0, [0, 2]])
            fy, sy = (val.item()/scale for val in K[1, [1, 2]])
            
            K_scaled = torch.tensor(
                [[1/scale, 0 , 0], [0, 1/scale, 0], [0, 0, 1.0]], 
                dtype=torch.float32, device=self.device
            ) @ K

            gts_per_image = input["instances"]

            gt_classes = gts_per_image.gt_classes
            
            # Filter out irrelevant groundtruth
            fg_selection_mask = (gt_classes != -1) & (gt_classes < self.num_classes)

            gt_classes = gt_classes[fg_selection_mask]
            gt_class_names = [self.thing_classes[cls_idx] for cls_idx in gt_classes]
            gt_boxes   = gts_per_image.gt_boxes.tensor[fg_selection_mask]  # 2D boxes
            gt_poses   = gts_per_image.gt_poses[fg_selection_mask]         # GT poses

            # projected 2D center, depth, w, h, l, 3D center
            gt_boxes3D = gts_per_image.gt_boxes3D[fg_selection_mask]

            # this box may have been mirrored and scaled so
            # we need to recompute XYZ in 3D by backprojecting.
            gt_z = gt_boxes3D[:, 2]

            gt_x3D = gt_z * (gt_boxes3D[:, 0] - sx)/fx
            gt_y3D = gt_z * (gt_boxes3D[:, 1] - sy)/fy
            
            # put together the GT boxes
            gt_center_3D = torch.stack((gt_x3D, gt_y3D, gt_z)).T
            gt_boxes3D_XYZ_WHL = torch.cat((gt_center_3D, gt_boxes3D[:, 3:6]), dim=1)

            gt_colors = torch.tensor(
                [util.get_color(i) for i in range(len(gt_boxes3D_XYZ_WHL))], 
                device=self.device
            )/255.0

            gt_meshes = util.mesh_cuboid(gt_boxes3D_XYZ_WHL, gt_poses, gt_colors)

            # perform a simple NMS, which is not cls dependent. 
            keep = batched_nms(
                instances_i.pred_boxes.tensor, 
                instances_i.scores, 
                torch.zeros(len(instances_i.scores), dtype=torch.long, device=instances_i.scores.device), 
                self.roi_heads.box_predictor.test_nms_thresh
            )
            
            keep = keep[:max_vis_prop]
            num_to_visualize = len(keep)

            pred_xyzwhl = torch.cat((instances_i.pred_center_cam[keep], instances_i.pred_dimensions[keep]), dim=1)
            pred_pose = instances_i.pred_pose[keep]

            pred_colors = torch.tensor(
                [util.get_color(i) for i in range(num_to_visualize)], 
                device=self.device
            )/255.0

            pred_boxes = instances_i.pred_boxes[keep]
            pred_scores = instances_i.scores[keep]
            pred_classes = instances_i.pred_classes[keep]
            pred_class_names = ['{} {:.2f}'.format(self.thing_classes[cls_idx], score) for cls_idx, score in zip(pred_classes, pred_scores)]
            pred_meshes = util.mesh_cuboid(pred_xyzwhl, pred_pose, pred_colors)

            # convert to lists
            pred_meshes = [pred_meshes.__getitem__(i).detach() for i in range(len(pred_meshes))]
            gt_meshes = [gt_meshes.__getitem__(i) for i in range(len(gt_meshes))]

            img_3DPR = vis.draw_scene_view(img_3DPR, K_scaled.cpu().numpy(), pred_meshes, text=pred_class_names, mode='front', blend_weight=0.0, blend_weight_overlay=0.85)
            img_3DGT = vis.draw_scene_view(img_3DGT, K_scaled.cpu().numpy(), gt_meshes, text=gt_class_names, mode='front', blend_weight=0.0, blend_weight_overlay=0.85)

            # horizontal stack 3D GT and pred left/right
            vis_img_3d = np.concatenate((img_3DGT, img_3DPR), axis=1)
            vis_img_3d = vis_img_3d[:, :, [2, 1, 0]] # RGB
            vis_img_3d = vis_img_3d.astype(np.uint8).transpose(2, 0, 1)

            storage.put_image("Left: GT 3D cuboids; Right: Predicted 3D cuboids", vis_img_3d)

            break  # only visualize one image in a batch


@META_ARCH_REGISTRY.register()
class BoxNet(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        depth_model: nn.Module,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: tuple[float],
        pixel_std: tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.depth_model = depth_model
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
    
    @classmethod
    def from_config(cls, cfg, priors=None):
        backbone = build_backbone(cfg, priors=priors)
        depth_model = 'zoedepth'
        pretrained_resource = 'local::depth/checkpoints/depth_anything_metric_depth_indoor.pt'
        d_model = setup_depth_model(depth_model, pretrained_resource) #NOTE maybe make the depth model be learnable as well
        return {
            "backbone": backbone,
            "depth_model": d_model,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape(), priors=priors),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }
            
    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]], normalise=True, img_type="image", convert=False, NoOp=False, to_float=False):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x[img_type]) for x in batched_inputs]
        if normalise:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        else:
            if convert:
                # convert from BGR to RGB
                images = [x[[2,1,0],:,:] for x in images]
            if to_float:
                images = [x.float()/255.0 for x in images]
            if NoOp:
                images = ImageList.from_tensors(images,0,)
                return images
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], segmentor, experiment_type, proposal_function):
        if not self.training:
            if not experiment_type['use_pred_boxes']: # MABO
                return self.inference(batched_inputs, do_postprocess=False, segmentor=segmentor, experiment_type=experiment_type, proposal_function=proposal_function)
            else: # AP
                return self.inference(batched_inputs, do_postprocess=True, segmentor=segmentor, experiment_type=experiment_type, proposal_function=proposal_function)

        if self.training:
            images = self.preprocess_image(batched_inputs, img_type='image', convert=False)
            images_raw = self.preprocess_image(batched_inputs, img_type='image', convert=True, normalise=False, NoOp=True)
            depth_maps = self.preprocess_image(batched_inputs, img_type="depth_map", normalise=False, NoOp=True)
            if batched_inputs[0]['ground_map'] is not None:
                ground_maps = self.preprocess_image(batched_inputs, img_type="ground_map", normalise=False, NoOp=True)
            else:
                #logger.info("ground map file not found, setting to None")
                ground_maps = None
            # scaling factor for the sample relative to its original scale
            # e.g., how much has the image been upsampled by? or downsampled?
            im_scales_ratio = [info['height'] / im.shape[1] for (info, im) in zip(batched_inputs, images)]
            # The unmodified intrinsics for the image
            Ks = [torch.FloatTensor(info['K']) for info in batched_inputs]

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            # the backbone is actually a FPN, where the DLA model is the bottom-up structure.
            # FPN: https://arxiv.org/abs/1612.03144v2
            # backbone and proposal generator only work on 2D images and annotations.
            features = self.backbone(images.tensor)
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)

            # images_raw are normalised to [0,1] and not resized here
            pred_o = self.depth_model(images_raw.tensor.float()/255.0)
            d_features = pred_o['depth_features']
            path_1 = d_features['path_1']
            p1 = F.interpolate(path_1, size=features['p2'].shape[-2:], mode='bilinear', align_corners=False)

            combined_features = torch.cat((features['p2'], p1), dim=1)
            print(gt_instances)
            results = self.roi_heads(images, images_raw, depth_maps, ground_maps, combined_features, proposals, Ks, im_scales_ratio, segmentor, experiment_type, targets=gt_instances[0].remove('gt_boxes3D')) # NOTE if batch > 1 then gt_instances[0] is wrong
            return results

    
    def inference(self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True, segmentor=None, experiment_type={}, proposal_function='propose'):
        assert not self.training

        # must apply the same preprocessing to both the image, the depth map, and the mask
        # except don't normalise the input for the segmentation method
        images = self.preprocess_image(batched_inputs, img_type='image', convert=False)
        images_raw = self.preprocess_image(batched_inputs, img_type='image', convert=True, normalise=False, NoOp=True)
        depth_maps = self.preprocess_image(batched_inputs, img_type="depth_map", normalise=False, NoOp=True)
        if batched_inputs[0]['ground_map'] is not None:
            ground_maps = self.preprocess_image(batched_inputs, img_type="ground_map", normalise=False, NoOp=True)
        else:
            #logger.info("ground map file not found, setting to None")
            ground_maps = None
            # TODO: make logic to predict ground map on the fly
            # logger.info("ground map file not found, computing...")
            # raise NotImplementedError("Implement ground on the fly, see generate_ground_segmentations.py for reference")

        # scaling factor for the sample relative to its original scale
        # e.g., how much has the image been upsampled by? or downsampled?
        im_scales_ratio = [info['height'] / im.shape[1] for (info, im) in zip(batched_inputs, images)]
        
        # The unmodified intrinsics for the image
        Ks = [torch.FloatTensor(info['K']) for info in batched_inputs]

        # do_postprocess is the same as using predicted boxes
        if do_postprocess:
            # gt_instances should be None in inference mode
            features = self.backbone(images.tensor)
            # normal inference
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            features, proposals = None, gt_instances

        # is it necessary to resize images back???

        # use the mask and the 2D box to predict the 3D box
        # proposals are ground truth for MABO plots and predictions for AP plots
        results = self.roi_heads(images, images_raw, depth_maps, ground_maps, features, proposals, Ks, im_scales_ratio, segmentor, experiment_type, proposal_function)
        return results
    
    def visualize_training(self, batched_inputs, proposals, instances):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
            instances (list): a list that contains predicted RoIhead instances. Both
                batched_inputs and proposals should have the same length.
        """
        
        storage = get_event_storage()

        # minimum number of boxes to try to visualize per image
        max_vis_prop = 20

        if not hasattr(self, 'thing_classes'):
            self.thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
            self.num_classes = len(self.thing_classes)

        for input, prop, instances_i in zip(batched_inputs, proposals, instances):

            img = input["image"]            
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            img_3DGT = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]]) # BGR
            img_3DPR = np.ascontiguousarray(img.copy()[:, :, [2, 1, 1]]) # BGR

            '''
            Visualize the 2D GT and proposal predictions
            '''
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img_rpn = np.concatenate((anno_img, prop_img), axis=1)
            vis_img_rpn = vis_img_rpn.transpose(2, 0, 1)
            storage.put_image("Left: GT 2D bounding boxes; Right: Predicted 2D proposals", vis_img_rpn)

            '''
            Visualize the 3D GT and predictions
            '''
            K = torch.tensor(input['K'], device=self.device)
            scale = input['height']/img.shape[0]
            fx, sx = (val.item()/scale for val in K[0, [0, 2]])
            fy, sy = (val.item()/scale for val in K[1, [1, 2]])
            
            K_scaled = torch.tensor(
                [[1/scale, 0 , 0], [0, 1/scale, 0], [0, 0, 1.0]], 
                dtype=torch.float32, device=self.device
            ) @ K

            gts_per_image = input["instances"]

            gt_classes = gts_per_image.gt_classes
            
            # Filter out irrelevant groundtruth
            fg_selection_mask = (gt_classes != -1) & (gt_classes < self.num_classes)

            gt_classes = gt_classes[fg_selection_mask]
            gt_class_names = [self.thing_classes[cls_idx] for cls_idx in gt_classes]
            gt_boxes   = gts_per_image.gt_boxes.tensor[fg_selection_mask]  # 2D boxes
            gt_poses   = gts_per_image.gt_poses[fg_selection_mask]         # GT poses

            # projected 2D center, depth, w, h, l, 3D center
            gt_boxes3D = gts_per_image.gt_boxes3D[fg_selection_mask]

            # this box may have been mirrored and scaled so
            # we need to recompute XYZ in 3D by backprojecting.
            gt_z = gt_boxes3D[:, 2]

            gt_x3D = gt_z * (gt_boxes3D[:, 0] - sx)/fx
            gt_y3D = gt_z * (gt_boxes3D[:, 1] - sy)/fy
            
            # put together the GT boxes
            gt_center_3D = torch.stack((gt_x3D, gt_y3D, gt_z)).T
            gt_boxes3D_XYZ_WHL = torch.cat((gt_center_3D, gt_boxes3D[:, 3:6]), dim=1)

            gt_colors = torch.tensor(
                [util.get_color(i) for i in range(len(gt_boxes3D_XYZ_WHL))], 
                device=self.device
            )/255.0

            gt_meshes = util.mesh_cuboid(gt_boxes3D_XYZ_WHL, gt_poses, gt_colors)

            # perform a simple NMS, which is not cls dependent. 
            keep = batched_nms(
                instances_i.pred_boxes.tensor, 
                instances_i.scores, 
                torch.zeros(len(instances_i.scores), dtype=torch.long, device=instances_i.scores.device), 
                self.roi_heads.box_predictor.test_nms_thresh
            )
            
            keep = keep[:max_vis_prop]
            num_to_visualize = len(keep)

            pred_xyzwhl = torch.cat((instances_i.pred_center_cam[keep], instances_i.pred_dimensions[keep]), dim=1)
            pred_pose = instances_i.pred_pose[keep]

            pred_colors = torch.tensor(
                [util.get_color(i) for i in range(num_to_visualize)], 
                device=self.device
            )/255.0

            pred_boxes = instances_i.pred_boxes[keep]
            pred_scores = instances_i.scores[keep]
            pred_classes = instances_i.pred_classes[keep]
            pred_class_names = ['{} {:.2f}'.format(self.thing_classes[cls_idx], score) for cls_idx, score in zip(pred_classes, pred_scores)]
            pred_meshes = util.mesh_cuboid(pred_xyzwhl, pred_pose, pred_colors)

            # convert to lists
            pred_meshes = [pred_meshes.__getitem__(i).detach() for i in range(len(pred_meshes))]
            gt_meshes = [gt_meshes.__getitem__(i) for i in range(len(gt_meshes))]

            img_3DPR = vis.draw_scene_view(img_3DPR, K_scaled.cpu().numpy(), pred_meshes, text=pred_class_names, mode='front', blend_weight=0.0, blend_weight_overlay=0.85)
            img_3DGT = vis.draw_scene_view(img_3DGT, K_scaled.cpu().numpy(), gt_meshes, text=gt_class_names, mode='front', blend_weight=0.0, blend_weight_overlay=0.85)

            # horizontal stack 3D GT and pred left/right
            vis_img_3d = np.concatenate((img_3DGT, img_3DPR), axis=1)
            vis_img_3d = vis_img_3d[:, :, [2, 1, 0]] # RGB
            vis_img_3d = vis_img_3d.astype(np.uint8).transpose(2, 0, 1)

            storage.put_image("Left: GT 3D cuboids; Right: Predicted 3D cuboids", vis_img_3d)

            break 

def build_model(cfg, priors=None):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg, priors=priors)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model


def build_backbone(cfg, input_shape=None, priors=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape, priors)
    assert isinstance(backbone, Backbone)
    return backbone