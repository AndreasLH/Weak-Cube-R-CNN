
cd GroundingDINO/
pip install -e .
mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

cd sam-hq
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth -O sam_hq_vit_b.pth
cd ..

cd depth
./download_models.sh
cd ..

mkdir output
mkdir output/Baseline_sgd
mkdir output/weak_cube_r-cnn
wget https://huggingface.co/AndreasLH/Weak-Cube-R-CNN/blob/main/Baseline_sgd/model_final.pth -O output/Baseline_sgd/model_final.pth

wget https://huggingface.co/AndreasLH/Weak-Cube-R-CNN/blob/main/weak%20cube%20r-cnn/model_final.pth -O output/weak_cube_r-cnn/model_final.pth