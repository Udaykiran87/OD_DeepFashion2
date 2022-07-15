## CUDA and cuDNN version
```bash
CUDA: 11.2
cuDNN: 8.1
```

## Object detection for DeepFashion2 dataset
DeepFashion2 object detection using Detectron2.

## STEPS-
### STEP 00- Clone repository and chnage directory to OD_DeepFashion2 directory

```bash
git clone https://github.com/Udaykiran87/OD_DeepFashion2.git

cd OD_DeepFashion2
```
### STEP 01- Create a conda environment and install necessary packages after opening the repository in VSCODE

```bash
init_setup.sh
```
### STEP 02- Activate the environment
```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 03- Create necessary folder structures: Stage 01
```bash
python src/components/stage_01_folder_setup.py
```

### STEP 04- Install detectron2: Stage 02
```bash
python src/components/stage_01_install_detectron2.py
```