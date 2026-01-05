## FLOL-Detection: Real-Time Low-Light Enhancement with Object Detection
**Based on the original work "FLOL: Fast Baselines for Real-World Low-Light Enhancement" by** **[Juan C. Benito](https://scholar.google.com/citations?hl=en&user=f186MIUAAAAJ), [Daniel Feijoo](https://scholar.google.com/citations?hl=en&user=hqbPn4YAAAAJ), [Alvaro Garcia](https://scholar.google.com/citations?hl=en&user=c6SJPnMAAAAJ), [Marcos V. Conde](https://scholar.google.com/citations?user=NtB1kjYAAAAJ&hl=en)** (CIDAUT AI  and University of Würzburg)

**Project Update:** This repository extends the original FLOL framework, which focused on single-image enhancement, to support real-time video processing and webcam streams. Additionally, we integrated a YOLO11n object detection model to perform detection on the enhanced low-light frames. This allows for robust object detection even in dark environments where standard detectors typically fail.

🚀 **New Features**
- Video Inference: Enhance low-light videos frame-by-frame with high temporal consistency.
- Webcam Demo: Real-time enhancement and detection using your computer's webcam.
- Object Detection: Integrated YOLO11n model to detect objects directly on enhanced frames.

| <img src="images/teaser/425_UHD_LL.JPG" alt="add" width="450"> | <img src="images/teaser/425_UHD_LL.png" alt="add" width="450"> | <img src="images/teaser/425_FLOL+.JPG" alt="add" width="450"> |
|:-------------------------:|:-------------------------:|:-------------------------:|
| Input              | UHDFour                | **FLOL** (ours)    |
| <img src="images/teaser/1778_UHD_LL.JPG" alt="add" width="450"> | <img src="images/teaser/1778_UHD_LL.png" alt="add" width="450"> | <img src="images/teaser/1778_FLOL+.JPG" alt="add" width="450"> |
| Input                 | UHDFour    | **FLOL** (ours)                 |

## 🛠️ **Network Architecture**

![add](images/general-scheme.png)

## 📦  **Dependencies and Installation**

- Python == 3.10.12
- PyTorch == 2.1.0
- CUDA == 12.1
- Other required packages in `requirements.txt`

```
# Clone this repository
git clone https://github.com/vinhquyen-lee/FLOL-Detection.git
cd FLOL-Detection

# Create python environment and activate it
python3 -m venv venv_FLOL-Detection
source venv_FLOL-Detection/bin/activate

# Install python dependencies
pip install -r requirements.txt
```
## 💻 **Datasets**
The datasets used for training and/or evaluation are:

|Paired Datasets     | Sets of images | Source  |
| -----------| :---------------:|------|
|LOLv2-real        | 689 training pairs / 100 test pairs | [Google Drive](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view) |
|LOLv2-synth        | 900 training pairs / 100 test pairs | [Google Drive](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view) |
|UHD-LL          | 2000 training pairs / 150 test pairs |[UHD-LL](https://drive.google.com/drive/folders/1IneTwBsSiSSVXGoXQ9_hE1cO2d4Fd4DN)|                                                                                    |
|MIT-5k          | 5000 training pairs / 100 test pairs  |[MIT-5k](https://data.csail.mit.edu/graphics/fivek/)|  
|LSRW-Nikon | 3150 training pairs / 20 test pairs | [R2RNet](https://github.com/JianghaiSCU/R2RNet) |
|LSRW-Huawei | 2450 training pairs / 30 test pairs | [R2RNet](https://github.com/JianghaiSCU/R2RNet) |

|Unpaired Datasets     | Sets of images | Source  |
| -----------| :---------------:|------|
|BDD100k          | 100k video clips  |[BDD100k](https://dl.cv.ethz.ch/bdd100k/data/)|
|DarkFace          | 6000 images |[DarkFace](https://drive.google.com/file/d/10W3TDvEAlZfEt88hMxoEuRULr42bIV7s/view)|
|DICM | 69 images | [DICM](https://ieeexplore.ieee.org/document/6615961)|
|LIME | 10 images | [LIME](https://drive.google.com/file/d/1OvHuzPBZRBMDWV5AKI-TtIxPCYY8EW70/view)|
|MEF | 17 images | [MEF](https://ieeexplore.ieee.org/document/7120119) |
|NPE | 8 images | [NPE](https://ieeexplore.ieee.org/document/6512558) |
|VV | 24 images | [VV](https://drive.google.com/file/d/1OvHuzPBZRBMDWV5AKI-TtIxPCYY8EW70/view)|

You can download LOLv2-Real and UHD-LL datasets and put them on the `/datasets` folder for testing. 

**Detection Dataset**
For the object detection task, the YOLO11n model was trained on a dataset:
[Obstacle Detection Computer Vision Dataset](https://universe.roboflow.com/aden-workspace/obstacle-detection-ttv7t)

## ✏️ **Results**
We present results in different datasets for FLOL+.

|Dataset     | PSNR| SSIM  | LPIPS|
|:-----------:|:------:|:------:|:------:|
|UHD-LL   | 25.01| 0.888| - |
|MIT-5k  | 22.10| 0.910|-|
|LOLv2-real | 21.75| 0.849|-|
|LOLv2-synth | 24.34| 0.906|-|
|LSRW-Both | 19.23| 0.583|0.273|

## ✈️ **Evaluation** 
To check our results you could run the evaluation of FLOL in each of the datasets:
- Run ```python evaluation.py --config ./options/LOLv2-Real.yml``` on your terminal to obtain PSNR and SSIM metrics. Default is UHD-LL.

- Run ```python lpips_metric.py  -g /LSRW_GroundTruthImages_path -p /LSRW_predictedimages -e .jpg``` on your terminal to obtain LPIPS value. (LSRW predicted images are obtained by using LOLv2-Real weight file)

**🎥 Video & Webcam Inference**
We provide new scripts to run inference on video files and live webcam streams.

**1. Video Inference**
To process a video file (enhance + detect), use the video_inference.py script:
- Run ``` python video_inference.py ```
 
**2. Webcam Demo**
To run real-time enhancement and detection on your webcam:
- Run ``` python webcam_demo.py ```
- 
Press q to quit the stream.

## 🚀 **Image Inference**
You can process the entire set of test images of provided datasets by running: 

- Run ```python inference.py --config ./options/LOLv2-Real.yml``` (UHD-LL is set by default)

Processed images will be saved in `./results/dataset_selected/`.

## 📷 **Gallery**

<p align="center"> <strong>  LSRW-Huawei </strong> </p>

| <img src="images/gallery/LSRW/Huawei/LSRWHuawei_input.png" alt="add" width="250"> | <img src="images/gallery/LSRW/Huawei/LSRWHuawei_fecnet.png" alt="add" width="250"> | <img src="images/gallery/LSRW/Huawei/LSRWHuawei_SNR.png" alt="add" width="250"> | <img src="images/gallery/LSRW/Huawei/LSRWHuawei_FourLLie.png" alt="add" width="250"> | <img src="images/gallery/LSRW/Huawei/LSRWHuawei_ours.png" alt="add" width="250"> |  <img src="images/gallery/LSRW/Huawei/LSRWHuawei_GT.png" alt="add" width="250"> |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| Input | FECNet |SNR-Net| FourLLIE | **FLOL** (ours) | Ground Truth|

<p align="center"> <strong>  LSRW-Nikon </strong> </p>

| <img src="images/gallery/LSRW/Nikon/LSRWN_input.png" alt="add" width="250"> | <img src="images/gallery/LSRW/Nikon/LSRWNikon_MIRNET.png" alt="add" width="250"> | <img src="images/gallery/LSRW/Nikon/LSRWN_RUAS.png" alt="add" width="250"> | <img src="images/gallery/LSRW/Nikon/LSRWN_EnGAN.png" alt="add" width="250"> | <img src="images/gallery/LSRW/Nikon/LSRWN_ours.png" alt="add" width="250"> |  <img src="images/gallery/LSRW/Nikon/LSRWN_GT.png" alt="add" width="250"> |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| Input | MIRNet |RUAS| EnGAN | **FLOL** (ours) | Ground Truth|


<p align="center"> <strong>  UHD-LL </strong> </p>

| <img src="images/gallery/UHD_LL/674_INPUT.JPG" alt="add" width="250"> | <img src="images/gallery/UHD_LL/674_UHDFour.png" alt="add" width="250"> | <img src="images/gallery/UHD_LL/674_FLOL.JPG" alt="add" width="250"> | <img src="images/gallery/UHD_LL/674_GT.JPG" alt="add" width="250"> |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| <img src="images/gallery/UHD_LL/1791_INPUT.JPG" alt="add" width="250"> | <img src="images/gallery/UHD_LL/1791_UHDFour.png" alt="add" width="250"> | <img src="images/gallery/UHD_LL/1791_FLOL.JPG" alt="add" width="250"> | <img src="images/gallery/UHD_LL/1791_GT.JPG" alt="add" width="250"> |
| Input | UHDFour | **FLOL** (ours) | Ground Truth|



