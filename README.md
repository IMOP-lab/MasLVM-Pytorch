# Multi-Aspect Fusion in Foundational Large Vision Model for Visible Light Medical Imaging Segmentation
This repository is the official implementation of MasLVM
## [Project page](https://github.com/IMOP-lab/MasLVM-Pytorch)

**Multi-Aspect Fusion in Foundational Large Vision Model for Visible Light Medical Imaging Segmentation.**

### Network structure
<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/MasLVM.png"width=100% height=100%>
</div>
<p align=center>
  Figure 1: Schematic diagram of MasLVM, including an SCE branch, an SSE branch, an HDME branch, and a Multi-Attention KAN Decoder. The 2D input image is first processed through Tri-Path encoder to extract three corresponding types of features. In mKAN, these features are simultaneously fed into the KAN multiple self-attention and iAFF modules for fusion to produce the output.
</p>

**We propose the MasLVM system, by encompassing extensive pretraining on both natural and medical imaging datasets to facilitate broad adaptability. The architecture integrates parallel semantic, spectral, and geometric feature processing pathways, enhancing multi-perspective information encoding. Evaluations across six standard benchmark datasets consistently exhibit MasLVM's advancements in segmentation performance compared to contemporary methods.**

#### Semantic Context Encoder
<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/SCE.png"width=50% height=50%>
</div>
<p align=center>
  Figure 2: The SCE commences the 2D input image through a patch embedding layer, followed by 16 Transformer layers. The feature output from the fourth layer is taken as the intermediate embedding, while the output from the final layer is taken as the image embedding. These two features are then plus to obtain the SCE feature. "pos_embed" presents the position embedding. "patch_embed" presents the patch embedding.
</p>

#### Spectral Spline Encoder
<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/SSE.png"width=50% height=50%>
</div>
<p align=center>
  Figure 3: The SSE initially processes the input image through a convolutional layer, followed by a series of MFFM, max pooling convolutions, and KAN Channel Attention, alternating through these layers. Post MFFM, two sequential convolutional layers generate SSE Features. The schematic of the MFFM contains two inputs: the feature after the conv layer and the low-frequency feature, both are weighted and combined followed by the IFFT to obtain the output.
</p>

#### Hierarchical Deformable Morphometry Encoder
<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/HDME.png"width=50% height=50%>
</div>
<p align=center>
  Figure 4: The HDME processes the input through convolutional blocks and max pooling layers, advancing into a dense module for feature aggregation, followed by additional convolutional layers, culminating in HDME Features.
</p>

## Experiment
### Baselines
**We have provided the GitHub links to the PyTorch implementation code for some networks compared to the experiments herein.**

[SwinUNETR](https://github.com/LeonidAlekseev/Swin-UNETR), [UNETR](https://github.com/tamasino52/UNETR), [ResUNet](https://github.com/rishikksh20/ResUnet)

**Segmentation results employing the isolated macular hole injection method, comparing the proposed MasLVM and prior segmentation models.**
<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/ISIC 2017.png"width=50% height=50%>
</div>
<p align=center>
  Figure 5: Comparative experimental results on the ISIC 2017 dataset.
</p>

<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/ISIC 2018.png"width=50% height=50%>
</div>
<p align=center>
  Figure 6: Comparative experimental results on the ISIC 2018 dataset.
</p>

<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/PH2.png"width=50% height=50%>
</div>
<p align=center>
  Figure 7: Comparative experimental results on the PH2 dataset.
</p>

<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/CVC.png"width=50% height=50%>
</div>
<p align=center>
  Figure 8: Comparative experimental results on the CVC-ClinicDB dataset.
</p>

<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/SEG.png"width=50% height=50%>
</div>
<p align=center>
  Figure 9: Comparative experimental results on the Kvasir-SEG dataset.
</p>

<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/Poly.png"width=50% height=50%>
</div>
<p align=center>
  Figure 10: Comparative experimental results on the Polypgen dataset.
</p>

### Ablation Study

<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/Results.png"width=50% height=50%>
</div>
<p align=center>
  Figure 11:Visual segmentation performance.
</p>

**Ablation study results for the MasLVM. The first column shows the original images, followed by segmentation results from each configuration (labeled k, a, b, c, d, e, f, g, h, i, j) in order as follows: (a) No Pre-train, (b) No HDME, (c) No SSE, (d) No SSE and HDME, (e) No MFFM, (f) No KAN Channel Attention, (g) No MFFM and KAN Channel Attention, (h) No iAFF, (i) No KAN multiple self-attention, (j) No iAFF and KAN multiple self-attention and (k) our proposed MasLVM.**

<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/ablation1.png"width=50% height=50%>
</div>
<p align=center>
  Figure 12:Analysis experiments to evaluate the effectiveness and robustness of the Tri-Path encoder on the ISIC 2017.
</p>

<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/ablation2.png"width=50% height=50%>
</div>
<p align=center>
  Figure 13:Analysis experiments to evaluate the effectiveness of the KAN channel attention and MFFM in SSE on the ISIC 2017.
</p>

<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/ablation3.png"width=50% height=50%>
</div>
<p align=center>
  Figure 14:Analysis experiments to evaluate the effectiveness of the iAFF and KAN multiple self-attention in mKAN on the ISIC 2017

<div align=center>
  <img src="https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/images/ablation4.png"width=50% height=50%>
</div>
<p align=center>
  Figure 15:Visual segmentation performance.
</p>

**Illustrations of two separate ablation studies. In (a), Column I displays the original image, Column II shows the  intermediate feature map produced using only SCE, Column III presents the result with two encoder branches incorporating SCE and HDME, Column IV represents the output with SCE and SSE, and Column V shows the output with all three encoder branches combined. In (b), Column I is the original image, Column II shows the result without KAN Channel Attention and MFFM, Column III presents the output without MFFM, Column IV displays the  intermediate feature map displayed without KAN Channel Attention, and Column V includes both modules.**

## License
This project is licensed under the [MIT license](https://github.com/IMOP-lab/MasLVM-Pytorch/blob/master/LICENSE).

