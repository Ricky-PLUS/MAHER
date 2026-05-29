
# MAHER: Multi-Agent Framework for Spatial Reasoning via Agent Harness Engineering

## Install

1. Clone this repository
```bash
git clone https://github.com/Ricky-PLUS/MAHER.git
cd MAHER
```

2. Install Package
```Shell
conda create -n MAHER python=3.11 -y
conda activate MAHER
pip install --upgrade pip 
pip install -e .
```
Remaining dependencies: 
"[WildDet3D](https://github.com/allenai/WildDet3D)"
"[MoGeV2](https://github.com/microsoft/MoGe)"
"[SAM 3](https://github.com/facebookresearch/sam3)"

## Inferencing
If you want to build Generated Functions (Function Generation), run the following code:
```Shell
python evaluate.py --workflow rag_build --annotations-json annotations.json --image-pth images/
```
To evaluate MAHER (Function Application), run the following code:
```Shell
python evaluate.py --workflow vlm_agent
```