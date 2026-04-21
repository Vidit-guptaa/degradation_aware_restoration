<div align="center">

<br/>

# 🖼️ Degradation-Aware Restoration

### Restore noisy, hazy, and blurred images.

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MindSpore](https://img.shields.io/badge/MindSpore-DL%20Framework-red?style=for-the-badge)](https://mindspore.cn)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()

<br/>

---

</div>

## 📌 About

A computer vision project that detects the **type of image degradation** (noise, blur, haze) and applies the most suitable restoration technique — combining traditional filters with a custom deep learning model.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Vidit-guptaa/degradation_aware_restoration.git
cd degradation_aware_restoration
pip install -r requirements.txt
python main.py
```

> Place your image at `input/sample.jpg` before running.

---

## 🗂️ Structure

```
📦 degradation_aware_restoration
 ┣ 📄 main.py          — Classical filter pipeline
 ┣ 🧠 model.py         — Deep learning model (MindSpore)
 ┣ 📁 input/           — Input images
 ┣ 📁 output/          — Restored results
 ┣ 📁 DFPIR/           — Feature perturbation module
 ┗ 📁 Rehaze/          — Dehazing module
```

---

## ⚙️ What It Does

| Step | Operation | Tool |
|------|-----------|------|
| 1️⃣ | Add Gaussian noise to simulate degradation | NumPy |
| 2️⃣ | Apply Gaussian, Median & Bilateral filters | OpenCV |
| 3️⃣ | Detect edges using Sobel + Canny | OpenCV |
| 4️⃣ | Segment image using Otsu thresholding | OpenCV |
| 5️⃣ | Evaluate quality with PSNR & SSIM | scikit-image |
| 6️⃣ | Save all results + comparison grid | Matplotlib |

---

## 🧠 Deep Learning Model

`model.py` defines a **residual encoder-decoder** network built in MindSpore.  
It learns to map degraded images → clean images using 12 residual blocks.

```
[Input] → Encoder (3→16→64) → Bottleneck → Decoder (64→32→3) → [Output]
```

---

## 📊 Sample Output

```
Gaussian Filter  →  PSNR: 28.4 dB  |  SSIM: 0.83
Median Filter    →  PSNR: 29.1 dB  |  SSIM: 0.85
Bilateral Filter →  PSNR: 31.6 dB  |  SSIM: 0.90  ✅ Best
```

Results are saved in `output/` including a **3×3 visual grid** (`all_results.png`).

---

## 👥 Contributors

<div align="center">

| Name | Roll No. |
|------|----------|
| **Vidit Gupta** | BT23ECI020 |
| **Dhairya Rathore** | BT23ECE038 |
| **Vaibhav Chouksey** | BT23ECE051 |
| **Saurabh Singh** | BT23ECE002 |

*IIIT NAGPUR — B.Tech ECE / EE — 2023–2027*

</div>

---

<div align="center">



</div>
