# DFGNet
# DISENTANGLED FEATURE-GUIDED MULTI-EXPOSURE HIGH DYNAMIC RANGE IMAGING (ICASSP 2022)

Keuntek Lee, Yeong Il Jang and Nam Ik Cho

VDID [[Arxiv](https://arxiv.org/abs/2104.00965)]

## Environments
- Ubuntu 18.04
- Pytorch 1.10.1
- CUDA 10.2
- CuDNN 7.6.5
- Python 3.8.3

## Abstract (DFGNet)

Multi-exposure high dynamic range (HDR) imaging aims to generate an HDR image from multiple differently exposed low dynamic range (LDR) images. It is a challenging task due to two major problems: (1) there are usually misalignments among the input LDR images, and (2) LDR images often have incomplete information due to under-/over-exposure. In this paper, we propose a disentangled feature-guided HDR network (DFGNet) to alleviate the above-stated problems. Specifically, we first extract and disentangle exposure features and spatial features of input LDR images. Then, we process these features through the proposed DFG modules, which produce a high-quality HDR image. Experiments show that the proposed DFGNet achieves outstanding performance on a benchmark dataset.
<br><br>

## Brief Description of Our Proposed Method

### <u>Overall Network Architecture</u>

<p align="center"><img src="figure/Overall.png" width="700"></p>

The overall architecture of proposed VDID, where <i><b>k</b></i>, <b><i>s</i></b>, and <i><b>n</i></b> denote kernel size, stride, and the number of filters.

## Experimental Results

**Results on Additive White Gaussian Noise (AWGN)**

<p align="center"><img src="figure/result_AWGN.png" width="900"></p>

**Results on Real-Noise (SIDD/DND)**

<p align="center"><img src="figure/result_REAL.png" width="900"></p>


### Test

[Options]
```
python main.py --gpu [GPU_number] --model [Type of model] --inputpath [dataset path] --dataset [dataset name] --sigma [noise level]  --noisy

--gpu: If you have more than one gpu in your computer, the number designates the index of GPU which is going to be used. [Default 0]
--model: Type of pretrained model for the test. Choices: ['model-AWGN', 'model-Real_Noise']. [Default: 'model-AWGN']
--inputpath: Path of input images [Default: dataset]
--dataset: Name of the dataset [Default: Kodak24]
--sigma: Noise level (Effective only for AWGN) [Default: 10]

--noisy: A flag whether input images are clean images or noisy images.
	-> input as a clean image to synthesize a noisy image which will be fed to the network.
	-> input as a noisy image directly fed to the network.
```

## Citation
```
@article{soh2021variational,
  title={Variational Deep Image Denoising},
  author={Soh, Jae Woong and Cho, Nam Ik},
  journal={arXiv preprint arXiv:2104.00965},
  year={2021}
}
```
