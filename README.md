# Agri-Tech Image Super-Resolution Model

This repository contains a deep learning solution designed to perform a blind 4x Super-Resolution upscale on crop leaf images. In modern smart agriculture, automated drones and mobile sensors frequently capture low-resolution, noisy, or highly compressed images. This project addresses severe information loss and JPEG artifacts by training an architecture to denoise and upscale inputs simultaneously.

The model reconstructs critical high-frequency biological textures—such as leaf veins, chlorosis, and necrotic lesions—transforming degraded 32x32 pixel inputs into mathematically faithful 128x128 pixel outputs.

---

## 🛠️ Core Constraints & Specifications

- **Architectural Constraints**: The networks are initialized completely at random. The use of pre-trained weights for the core generator/discriminator architectures is strictly prohibited to ensure authentic texture learning.
- **Leaderboard Evaluation Metric**: Models are evaluated using pixel-by-pixel Mean Absolute Error (MAE) across the three RGB channels. This minimization metric strictly penalizes spatial inaccuracies, preventing the network from generating realistic but hallucinated symptoms in the wrong coordinates.
- **Environment**: Configured to run entirely offline with internet access disabled and zero external data dependencies.

---

## 📐 Loss Functions

To optimize both visual realism and structural fidelity, the training framework utilizes a multi-component loss strategy:
1. **Adversarial & Perceptual Losses**: Trains the GAN to synthesize highly realistic textures. Perceptual loss is computed locally utilizing provided offline VGG-19 weights (`vgg19_weights.pth`).
2. **Mean Absolute Error (MAE)**: Minimizes the absolute difference between predicted and ground-truth pixel intensities to secure strict spatial accuracy.

---

