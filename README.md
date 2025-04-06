# Weak Cube R-CNN: Weakly Supervised 3D Object Detection using only 2D Bounding Box Annotations
Based on the Omni3D dataset & Cube R-CNN model 
[[`Project Page`](https://garrickbrazil.com/omni3d)] [[`arXiv`](https://arxiv.org/abs/2207.10660)] [[`BibTeX`](#citing)]

## Abstract 
> Monocular 3D object detection is an essential task in computer vision, and it has several applications in robotics and virtual reality. However, 3D object detectors are typically trained in a fully supervised way, relying extensively on 3D labeled data, which is labor-intensive and costly to annotate. This work focuses on weakly-supervised 3D detection to reduce data needs using a monocular method that leverages a single-camera system over expensive LiDAR sensors or multi-camera setups. We propose a general model Weak Cube R-CNN, which can predict objects in 3D at inference time, requiring only 2D box annotations for training by exploiting the relationship between 2D projections of 3D cubes.
Our proposed method utilizes pre-trained frozen foundation 2D models to estimate depth and orientation information on a training set. We use these estimated values as pseudo-ground truths during training. We design loss functions that avoid 3D labels by incorporating information from the external models into the loss. In this way, we aim to implicitly transfer knowledge from these large foundation 2D models without having access to 3D bounding box annotations.  
Experimental results on the SUN RGB-D dataset show increased performance in accuracy compared to an annotation time equalized Cube R-CNN omni3d baseline. While not precise for centimetre-level measurements, this method provides a strong foundation for further research.

Download the model on [**HuggingFace 🤗**](https://huggingface.co/AndreasLH/Weak-Cube-R-CNN) 

A **HuggingFace 🤗 demo** is now [live](https://huggingface.co/spaces/AndreasLH/Weakly-Supervised-3DOD) 


# Brief outline of method
We use rely heavily on the [Depth Anything]([github.com/](https://github.com/LiheYoung/Depth-Anything)) for processing the depth of images. An example of which can be seen below. 
<p align="center">
  <img src=".github/depth_map.png" />
</p>
We interpret the depth maps as point clouds like the one below. This enables us to identify the ground and determine which way is up in images.
<p align="center">
  <img src=".github/point_cloud_floor.png" />
</p>

# Example results

</table>
<table style="border-collapse: collapse; border: none;">
<tr>
	<td width="60%">
		<p align="center">
			Weak Cube R-CNN
			<img src=".github/weak1.jpg" alt=""/ height=>
		</p>
	</td>
	<td width="35%">
		<p align="center">
			Top view
			<img src=".github/weak2.jpg" alt="COCO demo"/ height=>
		</p>
	</td>
</tr>
</table>
<table style="border-collapse: collapse; border: none;">
<tr>
	<td width="60%">
		<p align="center">
			Time equalised Cube R-CNN
			<img src=".github/time1.jpg" alt=""/ height=>
		</p>
	</td>
	<td width="35%">
		<p align="center">
			Top view
			<img src=".github/time2.jpg" alt="COCO demo"/ height=>
		</p>
	</td>
</tr>
</table>
<table style="border-collapse: collapse; border: none;">
<tr>
	<td width="60%">
		<p align="center">
			Standard (benchmark) Cube R-CNN
			<img src=".github/cube1.jpg" alt=""/ height=>
		</p>
	</td>
	<td width="35%">
		<p align="center">
			Top view
			<img src=".github/cube2.jpg" alt="COCO demo"/ height=>
		</p>
	</td>
</tr>
</table>


# Results

| | **AP2D**    | **AP3D**    | **AP3D@15** | **AP3D@25**   | **AP3D@50**   |
|-----|----:|----:|----:|----:|----:|
| Weak Cube R-CNN | **12.62** | 4.88 | 8.44 | 3.77 | 0.06 |
| Time-equalised Cube R-CNN | 3.89 | 3.27 | 5.30 | 3.28 | **0.39** |
| Cube R-CNN | 16.51 | 15.08 | 21.34   | 16.2   | 4.56 |

# How the code works
All the models are defined in the different config files. The config files define different meta architectures and ROI heads. A meta architecture is the overall model, while the ROI head is the part doing the processing for each Region of interest. In theory these different components should be possible to swap between each method, such that for instance the weak loss uses the meta architecture of the proposal method, but this is not tested. 

It is possible to train only the 2D detector by using the Base_Omni3D.yaml and setting `MODEL.LOSS_W_3D = 0`. Check out the `.vscode/launch.json` for how to call the different experiments.


## Installation <a name="installation"></a>
Main dependencies
- Detectron2
- PyTorch
- PyTorch3D
- COCO
  
Rememmber to get all submodules
```sh
# to get the submodules
git submodule update --init
```

download all external models by running 
```sh
# to get models, including those trained by us
./download_models.sh
```

next, download the dataset as described in `DATA.md`.


We found it to be a bit finicky to compile Pytorc3D, so we would advise to pick a python version which has a prebuilt pytorch3D available (check their github). Otherwise this worked for us. We Use python 3.10 because then you wont have to build pytorch3d from source, which apparently does not work on the hpc
Detectron2. First install pytorch, load a corresponding cuda version that matches (check with a python torch.__version__, I used 12.1.1), then load a recent gcc version (>12.3)

```bash
pip install wheel ninja
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Pytorch3d
```bash
module swap nvhpc/23.5-nompi # get the cuda compiler
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html
```

We used both pytorch 2.0.1 and 2.1 for experiments

``` bash
# setup new evironment
conda create -n cubercnn python=3.10
source activate cubercnn

# main dependencies
conda install -c fvcore -c iopath -c conda-forge -c pytorch3d -c pytorch fvcore iopath pytorch3d pytorch torchvision cudatoolkit

# OpenCV, COCO, detectron2
pip install cython opencv-python
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# other dependencies
conda install -c conda-forge scipy seaborn
```

## Demo <a name="demo"></a>

To run the Cube R-CNN demo on a folder of input images using our `DLA34` model trained on the full Omni3D dataset,

``` bash
# Download example COCO images
sh demo/download_demo_COCO_images.sh

# Run an demo on our model
python demo/demo.py \
--config-file configs/Omni_combined.yaml \
--input-folder "datasets/coco_examples" \
--threshold 0.25 --display \
MODEL.WEIGHTS output/weak_cube_r-cnn/model_final.pth \
OUTPUT_DIR output/demo 
```
```bash
# Run an example of cube rcnn (baseline)
python demo/demo.py \
--config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
--input-folder "datasets/coco_examples" \
--threshold 0.25 --display \
MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \
OUTPUT_DIR output/demo 
```

See [`demo.py`](demo/demo.py) for more details. For example, if you know the camera intrinsics you may input them as arguments with the convention `--focal-length <float>` and `--principal-point <float> <float>`. See our [`MODEL_ZOO.md`](MODEL_ZOO.md) for more model checkpoints. 

## Training
In order to train the model successfully, some prerequisites are necessary.

1. Downloaded external models and data
2. Run cubercnn/data/generate_depth_maps.py
3. Run cubercnn/data/generate_ground_segmentations.py
4. Run the model in 2D-only mode first, until it is converged. (do this by setting LOSS_W_3D: 0.0 in the config file)
5. now you are ready to run the below

take a look at `submit.sh`
```bash
python tools/train_net.py \
    --resume \
    --config-file configs/Omni_combined.yaml \
    OUTPUT_DIR output/weak_cube-rcnn \
    log True \
    loss_functions "['iou', 'z_pseudo_gt_center', 'pose_alignment', 'pose_ground']" \
    MODEL.WEIGHTS output/omni3d-2d-only/model_final.pth \
    MODEL.ROI_CUBE_HEAD.LOSS_W_IOU 4.0 \
    MODEL.ROI_CUBE_HEAD.LOSS_W_NORMAL_VEC 40.0 \
    MODEL.ROI_CUBE_HEAD.LOSS_W_Z 100.0 \
    MODEL.ROI_CUBE_HEAD.LOSS_W_DIMS 0.1 \
    MODEL.ROI_CUBE_HEAD.LOSS_W_POSE 4.0 \
```

## Omni3D Data <a name="data"></a>
See [`DATA.md`](DATA.md) for instructions on how to download and set up images and annotations for training and evaluating Cube R-CNN. We only used the SUNRGBD dataset, but it is very easy to extend to other datasets listed in the datasets folder.

## (Excerpt from Omni3D on Training Cube R-CNN on Omni3D) <a name="training"></a>

We provide config files for trainin Cube R-CNN on
* Omni3D: [`configs/Base_Omni3D.yaml`](configs/Base_Omni3D.yaml)
* Omni3D indoor: [`configs/Base_Omni3D_in.yaml`](configs/Base_Omni3D_in.yaml)
* Omni3D outdoor: [`configs/Base_Omni3D_out.yaml`](configs/Base_Omni3D_out.yaml)

We train on 48 GPUs using [submitit](https://github.com/facebookincubator/submitit) which wraps the following training command,
```bash
python tools/train_net.py \
  --config-file configs/Base_Omni3D.yaml \
  OUTPUT_DIR output/omni3d_example_run
```

Note that our provided configs specify hyperparameters tuned for 48 GPUs. You could train on 1 GPU (though with no guarantee of reaching the final performance) as follows,
``` bash
python tools/train_net.py \
  --config-file configs/Base_Omni3D.yaml --num-gpus 1 \
  SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0025 \
  SOLVER.MAX_ITER 5568000 SOLVER.STEPS (3340800, 4454400) \
  SOLVER.WARMUP_ITERS 174000 TEST.EVAL_PERIOD 1392000 \
  VIS_PERIOD 111360 OUTPUT_DIR output/omni3d_example_run
```

### Tips for Tuning Hyperparameters <a name="tuning"></a>

Our Omni3D configs are designed for multi-node training. 

We follow a simple scaling rule for adjusting to different system configurations. We find that 16GB GPUs (e.g. V100s) can hold 4 images per batch when training with a `DLA34` backbone. If $g$ is the number of GPUs, then the number of images per batch is $b = 4g$. Let's define $r$ to be the ratio between the recommended batch size $b_0$ and the actual batch size $b$, namely $r = b_0 / b$. The values for $b_0$ can be found in the configs. For instance, for the full Omni3D training $b_0 = 196$ as shown [here](https://github.com/facebookresearch/omni3d/blob/main/configs/Base_Omni3D.yaml#L4).
We scale the following hyperparameters as follows:

  * `SOLVER.IMS_PER_BATCH` $=b$
  * `SOLVER.BASE_LR` $/=r$
  * `SOLVER.MAX_ITER`  $*=r$
  * `SOLVER.STEPS`  $*=r$
  * `SOLVER.WARMUP_ITERS` $*=r$
  * `TEST.EVAL_PERIOD` $*=r$
  * `VIS_PERIOD`  $*=r$

We tune the number of GPUs $g$ such that `SOLVER.MAX_ITER` is in a range between about 90 - 120k iterations. We cannot guarantee that all GPU configurations perform the same. We expect noticeable performance differences at extreme ends of resources (e.g. when using 1 GPU).

## Inference on Omni3D <a name="inference"></a>

To evaluate trained models from Cube R-CNN's [`MODEL_ZOO.md`](MODEL_ZOO.md), run

```
python tools/train_net.py \
  --eval-only --config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
  MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \
  OUTPUT_DIR output/evaluation
```

Our evaluation is similar to COCO evaluation and uses $IoU_{3D}$ (from [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/iou_box3d.py)) as a metric. We compute the aggregate 3D performance averaged across categories. 

To run the evaluation on your own models outside of the Cube R-CNN evaluation loop, we recommending using the `Omni3DEvaluationHelper` class from our [evaluation](https://github.com/facebookresearch/omni3d/blob/main/cubercnn/evaluation/omni3d_evaluation.py#L60-L88) similar to how it is utilized [here](https://github.com/facebookresearch/omni3d/blob/main/tools/train_net.py#L68-L114). 

The evaluator relies on the detectron2 MetadataCatalog for keeping track of category names and contiguous IDs. Hence, it is important to set these variables appropriately. 
```
# (list[str]) the category names in their contiguous order
MetadataCatalog.get('omni3d_model').thing_classes = ... 

# (dict[int: int]) the mapping from Omni3D category IDs to the contiguous order
MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id = ...
```

In summary, the evaluator expects a list of image-level predictions in the format of:
```
{
    "image_id": <int> the unique image identifier from Omni3D,
    "K": <np.array> 3x3 intrinsics matrix for the image,
    "width": <int> image width,
    "height": <int> image height,
    "instances": [
        {
            "image_id":  <int> the unique image identifier from Omni3D,
            "category_id": <int> the contiguous category prediction IDs, 
                which can be mapped from Omni3D's category ID's using
                MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
            "bbox": [float] 2D box as [x1, y1, x2, y2] used for IoU2D,
            "score": <float> the confidence score for the object,
            "depth": <float> the depth of the center of the object,
            "bbox3D": list[list[float]] 8x3 corner vertices used for IoU3D,
        }
        ...
    ]
}
```
