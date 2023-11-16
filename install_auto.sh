conda env list
which pip
pip install wandb
echo "[Finish] wandb"

# cuda_version=$(python -c "import torch; print(torch.version.cuda)")
# pytorch_version=$(python -c "import torch; print(torch.__version__)")
cuda_version="118"
pytorch_version="2.0.0"
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
url="https://data.pyg.org/whl/torch-${pytorch_version}+cu${cuda_version}.html"
pip install torch-scatter -f $url
echo "[Finish] torch-scatter"
pip install torch-sparse -f $url
echo "[Finish] torch-sparse"
pip install torch-cluster -f $url
echo "[Finish] torch-cluster"
pip install torch-spline-conv -f $url
echo "[Finish] torch-spline-conv"
pip install torch-geometric  -f $url
echo "[Finish] torch-geometric"
pip install fair-esm
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
echo "[Finish] dllogger"
pip install h5py pillow easydict modelcif hydra-core gpustat
echo "[Finish] h5py pillow easydict modelcif"
conda install -y biopython
echo "[Finish] biopython"
conda install -y easydict pyyaml -c conda-forge
echo "[Finish] easydict pyyaml"

cd ../AE_install
git clone https://github.com/aqlaboratory/openfold.git
cd openfold
python setup.py install

cd ..
git clone https://github.com/DeepGraphLearning/torchdrug.git
cd torchdrug
which pip
python setup.py develop
cd ../../Au*