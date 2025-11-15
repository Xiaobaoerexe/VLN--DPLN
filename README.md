# VLN-DPLN

## Content

- [Installation](#installation)
- [Preparation](#preparation)
- [VLN Training](#vln-training)
- [Acknowledgements](#acknowledgements)

## Installation

1. Navigate to the project directory:
```bash
cd VLN-DPLN
```

2. Create a Python 3.8.5 environment named `dpln`:
```bash
conda create -n dpln python=3.8.5
conda activate dpln
```

3. Install dependencies from the environment file:
```bash
conda env update -f environment.yml
```

**Note:** Please manually configure the accelerator settings according to your GPU count and server configuration.

## Preparation

### Training Datasets

1. Create a datasets directory under the VLN-DPLN path:
```bash
mkdir -p datasets
```

2. Follow our baseline [DUET](https://github.com/cshizhe/VLN-DUET) to configure the training datasets:
   - Visit the DUET homepage
   - Navigate to the Requirements section and click on the third item (Dropbox link)
   - Download the datasets and place them in the corresponding locations within the `datasets` folder

### Language Encoder

Download the language encoder from our [Google Drive](https://drive.google.com/file/d/1B097BWvUbeLzYXQ6ti1wWzCsfmuLMi-7/view?usp=sharing), extract it, and place the folder in `pretrain_src_dpln`.

**Note:** If the link is unavailable, please contact us via email.

### Visual Encoder

1. Download DUET's visual encoders while downloading the training datasets and place them in the corresponding locations.

2. Download the VLN-RAM visual encoder from [VLN-RAM](https://github.com/SaDil13/VLN-RAM):
   - Visit their homepage
   - Go to VLN Training section, item 2
   - Download from their Google Drive

3. Download the CLIP H14 encoder and env edit features from [ScaleVLN](https://github.com/wz0919/ScaleVLN):
   - Visit their homepage
   - Navigate to the R2R section, item 2
   - Download from Hugging Face

### Initialization Model

Download the pre-trained LXMERT model and place it in `datasets/pretrained/LXMERT`:
```
https://nlp.cs.unc.edu/data/model_LXRT.pth
```

### Our Pre-trained Weights

To reproduce our results, download our weights from [Google Drive](https://drive.google.com/file/d/1x4p2P90DOHWMSAi8YosdbiDIN57qT0JK/view?usp=sharing) and place them in `datasets/R2R/trained_models`.

**Note:** If the link is unavailable, please contact us via email.

## VLN Training

### Activate Environment
```bash
conda activate dpln
```

### Pre-training

For R2R, R4R, RxR datasets:
```bash
cd pretrain_src_dpln
bash run_r2r.sh
bash run_r4r.sh
bash run_rxr.sh
bash run_aug_h14_r2r.sh
```

For REVERIE dataset:
```bash
cd pretrain_src_dpln_rvr
bash run_aug_h14_reverie.sh
```

**Important:** Please check the configuration files before starting training.

### Pre-trained Model Validation (Optional)

We provide a script to batch validate pre-trained weights using multiple GPUs:

```bash
cd map_nav_src_dpln  # or cd map_nav_src_dpln_rvr
bash scripts/r2rtest.sh
```

To test other datasets, replace the sh file with other files containing the "test" keyword. Remember to update the batch weight paths in the sh files to your pre-trained weight save paths.

### Fine-tuning

```bash
cd map_nav_src_dpln  # or cd map_nav_src_dpln_rvr
bash scripts/r2rgpu0.sh
```

**Note:** We have prepared separate sh files for each GPU to enable simultaneous testing of multiple parameter configurations.

### Testing

To run testing:
1. Comment out the first two training stages in the sh file
2. Uncomment the last stage
3. Replace the model path with your own trained model path
4. Start testing

## Acknowledgements

We sincerely thank the following projects for their open-source contributions:

- [DUET](https://github.com/cshizhe/VLN-DUET)
- [EnvEdit](https://github.com/jialuli-luka/EnvEdit)
- [HAMT](https://github.com/cshizhe/VLN-HAMT)
- [VLN-RAM](https://github.com/SaDil13/VLN-RAM)
- [ScaleVLN](https://github.com/wz0919/ScaleVLN)

Their outstanding work has been invaluable to this project.
