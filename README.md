# Face2Parameter： A simple method for modeling facial parameters of game characters based on latent representation of VAE
![architecture](assets/arch.jpg)

## 0. Introduction
- The project aims to use deep learning methods to modeling facial parameters of game characters from images.
- This project takes the modeling of facial parameters of characters in Illusion's games as an example.

## 1. Installation
- Firstly, you need to clone this project using the following command:
```bash
git clone https://github.com/ChasonJiang/Face2Parameter.git
```
- Then, you need to first install the following dependencies:
	- Python 3.8+ 
	- Pytorch
	- Numpy 
	- Opencv
	- Tensorboard
- You can use the following installation command:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy
pip install opencv-python
pip install tensorboard
```
- Recommend: It is recommended to use anaconda to install in a virtual environment

## 2. Train
- Training the model is divided into two stage
    - Stage 1: Train the VAE model for learning the latent representation of images. 
  
        ```python
        python vae_trainer.py
        ```
    - Stage 2: Train the F2P model for learning the facial parameters of game characters based on the latent representation of VAE.
  
        ```python
        python extract_latentvec.py # extract the latent representation of images
        python f2p_trainer.py
        ```
    - Note:
        - For the first stage:
          - The publicly available datasets for training VAE models are: Celebra、FFHQ. And the dataset [HS_FACE](https://pan.baidu.com/s/1yPftN5rmtY5QDF7G2RjN4A?pwd=p8qd) independently produced by this project.
          - After alignment, the resolution of each image is 256x256, You can run face_alignment.py for face alignment.
        - For the second stage:
          - At this stage, we only trained on the HS_FACE dataset.
          - Training too many epochs is meaningless. In experience, around 15-30 epochs are sufficient.

## About HS_FACE dataset
- The HS_SACE dataset is a collection of approximately 14w facial images of game characters. It consists of three parts: 1 Facial images of game characters directly sampled from the game; 2. Use stable diffusion and use the image in 1 as a condition to generate facial images that are close to real people; 3. Facial parameters of game characters in 1.