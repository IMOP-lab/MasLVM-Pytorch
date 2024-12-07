# Multi-Aspect Fusion in Foundational Large Vision Model for Visible Light Medical Imaging Segmentation

Multi-Aspect Fusion in Foundational Large Vision Model for Visible Light Medical Imaging Segmentation

Xingru Huang, Tianyun Zhang, Zhaoyang Xu, Jian Huang, Haopeng Huang, Han Yang, Binfeng Zou, Shouqin Ding, Zhao Huang, Huiyu Zhou, Jin Liu, Zhiwen Zheng, Shaowei Jiang, and Xiaoshuai Zhang

Hangzhou Dianzi University IMOP-lab

## Methods
### MasLVM
<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/main/images/MasLVM.png"width=70% height=70%>
</div>
<p align=center>
  Figure 1: Schematic diagram of MasLVM, including an SCE branch, an SSE branch, an HDME branch, and a Multi-Attention KAN Decoder. The 2D input image is first processed through Tri-Path encoder to extract three corresponding types of features. In mKAN, these features are simultaneously fed into the KAN multiple self-attention and iAFF modules for fusion to produce the output.
</p>

We propose the MasLVM system, by encompassing extensive pretraining on both natural and medical imaging datasets to facilitate broad adaptability. The architecture integrates parallel semantic, spectral, and geometric feature processing pathways, enhancing multi-perspective information encoding. Evaluations across six standard benchmark datasets consistently exhibit MasLVM's advancements in segmentation performance compared to contemporary methods.

#### Semantic Context Encoder
<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/main/images/SCE.png"width=70% height=70%>
</div>
<p align=center>
  Figure 2: The SCE commences the 2D input image through a patch embedding layer, followed by 16 Transformer layers. The feature output from the fourth layer is taken as the intermediate embedding, while the output from the final layer is taken as the image embedding. These two features are then plus to obtain the SCE feature. "pos_embed" presents the position embedding. "patch_embed" presents the patch embedding.
</p>

#### Spectral Spline Encoder
<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/main/images/SSE.png"width=70% height=70%>
</div>
<p align=center>
  Figure 3: The SSE initially processes the input image through a convolutional layer, followed by a series of MFFM, max pooling convolutions, and KAN Channel Attention, alternating through these layers. Post MFFM, two sequential convolutional layers generate SSE Features. The schematic of the MFFM contains two inputs: the feature after the conv layer and the low-frequency feature, both are weighted and combined followed by the IFFT to obtain the output.
</p>

#### Hierarchical Deformable Morphometry Encoder
<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/main/images/HDME.png"width=70% height=70%>
</div>
<p align=center>
  Figure 4: The HDME processes the input through convolutional blocks and max pooling layers, advancing into a dense module for feature aggregation, followed by additional convolutional layers, culminating in HDME Features.
</p>

## Experiment
