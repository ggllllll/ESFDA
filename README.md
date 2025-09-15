# :page_facing_up: Progressive Pseudo-labels Enhancement for Source-Free Domain Adaptation Medical Image Segmentation

<p align="center"><img src="figures/overview.png" width="90%"></p>

### Dependency Preparation

```shell
cd ESFDA_SFDA
# Python Preparation
conda create -n ESFDA_SFDA python=3.8.5
activate ESFDA_SFDA
# (torch 1.7.1+cu110) It is recommended to use the conda installation on the Pytorch website https://pytorch.org/
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

### Model Training and Inference


- 1. Download the dataset and modify the relevant paths in the configuration file.
- 2. Source Model Train
  -- We use the code provided by [ProSFDA](https://github.com/ShishuaiHu/ProSFDA) to train the source model. If you want to use our trained source model, please contact me.
- 3. Generation phase: Generate target domain pseudo-labels
```shell
          python generate_pseudo.py
```
- 4. Adaptation stage: the source model adapts to the target domain
```shell
          python Train_target.py
```
- 5. Validation stage: You can get the pre-trained weights at https://drive.google.com/file/d/1XOjpNu_SO40vKy820OYRpe5ZVgUtNubD/view?usp=drive_link and validate it.
```shell
          python Validate.py
```
