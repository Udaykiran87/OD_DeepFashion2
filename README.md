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
python src/components/stage_02_install_detectron2.py
```

### STEP 05- Download and prepare dataset: Stage 03
```bash
python src/components/stage_03_prepare_data.py
```

### STEP 06- Custom Training: Stage 04
```bash
python src/components/stage_04_custom_train.py
```

### STEP 05- Prediction on an image: Stage 05
```bash
python src/components/stage_05_predict.py --test-image "artifacts/workspace/images/test/image/000003.jpg"
```
