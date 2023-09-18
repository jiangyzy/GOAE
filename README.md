
# Make Encoder Great Again in 3D GAN Inversion through Geometry and Occlusion-Aware Encoding (ICCV 2023)




[Paper](https://arxiv.org/abs/2303.12326) | [Project page](https://eg3d-goae.github.io/) | [Demo video](https://www.youtube.com/watch?v=CptQDMqM9Pc)

 Official implementation of "Make Encoder Great Again in 3D GAN Inversion through Geometry and Occlusion-Aware Encoding" ICCV 2023

<div align="center">
<img src="assets/teaser.gif" >
</div>


## :fire: Introduction
We present a encoder-based 3D generative adversarial network (GAN) inversion framework that can efficiently synthesize photo-realistic novel views while preserving geometry and details of the input image.

<div align="center">
<img src="assets/framework.png" width="600px"/>  
</div>

## :desktop_computer: Requirements
* 64-bit Python 3.8 and PyTorch 1.11.0 (or later).
* CUDA toolkit 11.3 or later.  
* Python libraries: see [requirements.txt](./goae/requirements.txt)

```bash
cd goae
conda create --name goae python=3.8
conda activate goae
pip install -r requirements.txt
```

## :running_woman: Inference

### Prepare Dataset 
Dataset preparation can refer to [EG3D](https://github.com/NVlabs/eg3d/) or these [codes](https://github.com/FeiiYin/SPI/blob/main/preprocess/)

### Download Models
The pretrained model checkpoint can be downloaded from [google drive](https://drive.google.com/drive/folders/12pTX5TKQcA8ElNW5jDkWURSPUyISggHs?usp=sharing), Put those checkpoint into  the directory  `GOAE/pretrained` . Note that current pretrained AFA only modifies the triplane on 32*32 resolution, more higher resolution modify can achieve better result.

### Commands

You can use the command below to test the example.

```bash
python infer.py --multi_view --video
```

You can use the command below to edit the example.

```bash
python infer.py --multi_view --video --edit --edit_attr glass --alpha 1.0
```


## :handshake: Citation
If you find this work useful for your research, please cite:
```
@article{yuan2023make,
  title={Make Encoder Great Again in 3D GAN Inversion through Geometry and Occlusion-Aware Encoding},
  author={Yuan, Ziyang and Zhu, Yiming and Li, Yu and Liu, Hongyu and Yuan, Chun},
  journal={arXiv preprint arXiv:2303.12326},
  year={2023}
}
```
## :mailbox: Contact
If you have any comments or questions, please open a new issue or feel free to contact Ziyang Yuan (yuanzy22@mails.tsinghua.edu).
