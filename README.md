# AEGIS — Airspace Classification & Micro-Doppler CNN for DTU Innovation course August 2026

AEGIS is an integrated multi-sensor airspace surveillance and drone identification prototype designed for Copenhagen Airport (EKCH). This repository contains the real-time classification console (`aegis-console_v2.html`), the trained PyTorch neural network (`MicroDopplerCNN`), the signal pre-processing pipeline, and automated export tools for live radar replay.

---

## 1. Model Architecture: `MicroDopplerCNN`

The core classification model is **`MicroDopplerCNN`** (implemented in [`train_micro_doppler_cnn.py`](train_micro_doppler_cnn.py) and saved in [`micro_doppler_cnn.pth`](micro_doppler_cnn.pth)). It is a custom 2D convolutional neural network specifically engineered for processing Range-Doppler ($5 \times 256$) micro-Doppler radar signatures.

```
Input Range-Doppler Tensor: (Batch, 1, 5, 256)
  │
  ├── Stage 1: ConvBNAct(1 -> 24, k=(3,5), p=(1,2)) + MaxPool2d(1, 2)  --> Shape: (B, 24, 5, 128)
  │     └─ Preserves 5 range bins; downsamples Doppler axis
  │
  ├── Stage 2: ConvBNAct(24 -> 48, k=(3,5), p=(1,2)) + MaxPool2d(1, 2) --> Shape: (B, 48, 5, 64)
  │     └─ Deepens spectral feature maps; downsamples Doppler axis
  │
  ├── Stage 3: ConvBNAct(48 -> 96, k=(3,3), p=(1,1)) + MaxPool2d(2, 2) --> Shape: (B, 96, 2, 32)
  │     └─ First spatial range downsampling (5 -> 2)
  │
  ├── Stage 4: ConvBNAct(96 -> 128, k=(3,3), p=(1,1)) + MaxPool2d(2, 2) --> Shape: (B, 128, 1, 16)
  │     └─ Second spatial range downsampling (2 -> 1)
  │
  ├── Global Pooling: AdaptiveAvgPool2d((1, 1))                        --> Shape: (B, 128, 1, 1)
  ├── Flatten                                                          --> Shape: (B, 128)
  │
  └── Classifier Head:
        ├── Dropout(p=0.30)
        ├── Linear(128 -> 64)
        ├── GELU()
        ├── Dropout(p=0.15)
        └── Linear(64 -> 4)                                            --> Shape: (B, 4) Logits
```

### Layer Specification

| Layer / Block | Operation | In Channels / Dim | Out Channels / Dim | Kernel / Pool | Output Tensor Shape |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | Raw spectrogram | — | 1 | — | $(B, 1, 5, 256)$ |
| **Stage 1** | `Conv2d` + `BatchNorm` + `GELU` | 1 | 24 | $(3 \times 5)$, pad $(1, 2)$ | $(B, 24, 5, 256)$ |
| | `MaxPool2d` | 24 | 24 | stride $(1, 2)$ | $(B, 24, 5, 128)$ |
| **Stage 2** | `Conv2d` + `BatchNorm` + `GELU` | 24 | 48 | $(3 \times 5)$, pad $(1, 2)$ | $(B, 48, 5, 128)$ |
| | `MaxPool2d` | 48 | 48 | stride $(1, 2)$ | $(B, 48, 5, 64)$ |
| **Stage 3** | `Conv2d` + `BatchNorm` + `GELU` | 48 | 96 | $(3 \times 3)$, pad $(1, 1)$ | $(B, 96, 5, 64)$ |
| | `MaxPool2d` | 96 | 96 | stride $(2, 2)$ | $(B, 96, 2, 32)$ |
| **Stage 4** | `Conv2d` + `BatchNorm` + `GELU` | 96 | 128 | $(3 \times 3)$, pad $(1, 1)$ | $(B, 128, 2, 32)$ |
| | `MaxPool2d` | 128 | 128 | stride $(2, 2)$ | $(B, 128, 1, 16)$ |
| **Pool** | `AdaptiveAvgPool2d` | 128 | 128 | target $(1, 1)$ | $(B, 128, 1, 1)$ |
| **Dense 1** | `Dropout(0.3)` + `Linear` + `GELU` | 128 | 64 | — | $(B, 64)$ |
| **Dense 2** | `Dropout(0.15)` + `Linear` | 64 | 4 | — | $(B, 4)$ |

### Computational Characteristics
- **Total Parameters:** $\approx 178,617$ trainable weights (compact footprint).
- **Inference Latency:** $< 1.5\,\text{ms}$ on CPU, $< 0.2\,\text{ms}$ on Apple Silicon / MPS or CUDA GPU.
- **Activation Function:** `GELU` (Gaussian Error Linear Unit) used throughout for smooth gradients.

---

## 2. Target Classes & Discrimination

The network classifies radar targets into 4 distinct operational categories:

1. **`drone` (Class 0):** Multirotor and fixed-wing unmanned systems (DJI Matrice D1, Phantom D3, Mavic D6, etc.). Characterized by high-frequency periodic blade micro-Doppler harmonics ($\pm 50\text{--}80\,\text{Hz}$ spread).
2. **`bird` (Class 1):** Biological targets (seagulls, pigeons, ravens). Characterized by low-frequency sinusoidal wingbeat patterns ($\sim 5.4\,\text{Hz}$ envelope modulation).
3. **`clutter` (Class 2):** Static obstacles and corner reflectors (CR). Characterized by a sharp peak at zero-Doppler with narrow spectral spread ($< 10\,\text{Hz}$).
4. **`human` (Class 3):** Terrestrial movement (walking/running gait harmonics).

---

## 3. Signal Processing & Preprocessing Pipeline

Raw radar measurements originate from a 77 GHz FMCW radar dataset (`radar_data.npy`). Each measurement segment is transformed via the following pipeline:

1. **Matrix Reshape:** The 1280-element complex time-series vector is reshaped into $5\,\text{range bins} \times 256\,\text{slow-time azimuth sweeps}$.
2. **Slow-time Windowing:** A 256-point Hanning window $w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{255}\right)\right)$ is applied along the azimuth axis to suppress spectral leakage and sidelobes.
3. **Doppler FFT:** A 256-point FFT is computed across slow-time with zero-frequency centering (`fftshift`) to obtain the micro-Doppler spectrum.
4. **Log-Magnitude Conversion:** Converted to power in decibels:
   $$\text{RD}_{\text{dB}} = 20 \log_{10}(|\text{FFT}| + 10^{-12})$$
5. **Standardization:** Zero-mean unit-variance normalization:
   $$\text{RD}_{\text{norm}} = \frac{\text{RD}_{\text{dB}} - \mu}{\sigma + 10^{-8}}$$
6. **Spectral Metrics Extraction:**
   - **Doppler Centroid:** Radial velocity shift of target body ($\text{Hz}$).
   - **Doppler Spread:** Root-mean-square spectral width quantifying micro-motion ($\text{Hz}$).

---

## 4. Training & Data Augmentations

- **Loss Function:** `nn.CrossEntropyLoss(weight=class_weights)` with inverse class-frequency weighting to handle the class imbalance between drone flights and clutter samples.
- **Optimizer:** `AdamW` (learning rate $\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$).
- **Scheduler:** `CosineAnnealingLR` decaying over epochs to $\eta_{\text{min}} = 10^{-6}$.
- **Radar Domain Augmentations:**
  - **Complex AWGN Injection:** Simulated receiver thermal noise ($15\text{--}30\,\text{dB}$ SNR range).
  - **Random Doppler Shifts:** Rolling along the Doppler axis by $\pm 12\,\text{bins}$ to simulate variations in bulk radial velocity.
  - **Random Doppler Flips:** $50\%$ probability horizontal reversal to model inbound vs. outbound flight paths.

---

## 5. Console Demonstration Integration (`aegis-console_v2.html`)

For zero-dependency browser demonstration:
1. [`export_radar_scenarios.py`](export_radar_scenarios.py) runs [`MicroDopplerCNN`](train_micro_doppler_cnn.py) inference with [`micro_doppler_cnn.pth`](micro_doppler_cnn.pth) over real FMCW radar tracks from `radar_data.npy`.
2. Model predictions, softmax class probabilities, and spectral spread/centroid metrics are exported into [`radar_scenarios.js`](radar_scenarios.js).
3. [`aegis-console_v2.html`](aegis-console_v2.html) replays the scenario, dynamically updating:
   - Live multi-sensor fusion confidence.
   - Micro-Doppler waterfall and real-time FFT power envelope curves calibrated to the measured centroid drift and blade spread.
   - Airspace threat alerts, Remote ID correlation, and ATC escalation actions.

### Running the Export & Console

To re-export scenarios from the trained model:
```bash
python3 export_radar_scenarios.py
```

To view the console:
Open `aegis-console_v2.html` in any web browser.
