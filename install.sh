uv venv .venv python=3.10

source .venv/bin/activate

uv pip install cudatoolkit==11.7

uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

uv pip install torch-scatter==2.1.0+pt113cu117 torch-sparse==0.6.16+pt113cu117 torch-cluster==1.6.0+pt113cu117 torch-spline-conv==1.2.1+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

uv pip install torch-geometric==2.2.0

uv pip install PyYAML scipy "networkx[default]" biopython pandas click

uv pip install peft biotite numba ruamel.yaml billiard seaborn numba_progress

uv pip install datasets==2.14.4

uv pip install foldseek==8.ef4e960

git clone https://github.com/facebookresearch/esm

cd esm

uv pip install -e .

cd ..
