# Experiments with Denoising Diffusion Probabilistic Models

In this project I've implemented image generation method which is close to approach from [DDPM paper](https://arxiv.org/abs/2006.11239).

The main task is to experiment with different image generating methods using diffusion models and implement them from scratch.

I've started with basic DDPM model (without any improvements such as [nonlinear scedule](https://arxiv.org/abs/2102.09672) of $\beta_t$ or [non-Marcovian processes](https://arxiv.org/abs/2010.02502)).

Now I'm experimenting with [MNIST](http://yann.lecun.com/exdb/mnist/), [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Flickr-Faces-HQ](https://github.com/NVlabs/ffhq-dataset) datasets. The results of model training on the MNIST dataset are presented below.
## List of contents
* <b>MNIST_DDPM.ipynb</b> -- main notebook file with minimalistic implementation of diffusin model for MNIST dataset (you can use it to load checkpoint and generate samples or train model)
* checkpoints -- checkpoints for models
* images&samples -- some examples of generated images / animation


## Model trained on MNIST
Model uses $T=1000$ steps for 32x32 image generating. It was trained for 48 epochs on Google Colab.

### Generated samples
<p align="center">
  <img src="https://user-images.githubusercontent.com/58548935/178595300-0abb340a-80ae-4d3d-bcd3-568e309c2680.png" width="700">
  <img src="https://github.com/MatthewMih/small_projects/blob/main/Diffusion_models/images%26samples/48epMNIST.png" width="475">
  
</p>

### Generation process step-by-step:
<p align="center">
<img src="https://user-images.githubusercontent.com/58548935/178343762-86175901-ac4c-4e90-8165-1be5ca2d04fc.png" width="900">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/58548935/178343302-87de92ed-dc68-4074-ba16-07dee4cf0a6d.gif" width="450">
  <img src="https://github.com/MatthewMih/small_projects/blob/main/Diffusion_models/images%26samples/gifka_4x.gif" width="450">
</p>

## Other datasets
Now the quality of images generated on CelebA/FFHQ-trained models is too low. I'm conducting experiments and training new models to improve it.

<p align="center">
<img src="https://user-images.githubusercontent.com/58548935/178600767-8045b470-de21-45ca-a581-0bd7d6db39e4.png" width="450" title="CelebA">
</p>
