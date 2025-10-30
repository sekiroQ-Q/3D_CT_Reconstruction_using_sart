# 3D CT Reconstruction using SART (Simultaneous Algebraic Reconstruction Technique)

This project implements a **3D Computed Tomography (CT) reconstruction algorithm** using the **SART** (or **SIRT**) approach with the **ASTRA Toolbox**.  
It performs **slice-by-slice 2D reconstruction** from a set of X-ray projection images (`.tif`), reconstructing a full 3D volume.

---

## Features

- Slice-by-slice **2D fan-beam reconstruction** using **ASTRA CUDA backend** which result in faster inference
- **Automatic projection loading**, shape validation, and log-transform preprocessing.
- Configurable **geometric parameters** (SOD, SDD, voxel size, etc.).
- Saves reconstructed slices as `.tif` images.

---

## Project Structure

```
3D_CT_Reconstruction/
│
├── raw/
│   ├── projections/          # Input projection .tif images
│   ├── reconstruction/       # reconstructed slices given in data
│   └── reconstruction_out/   # Output reconstructed slices
│
├── 3d_reconstruct.py         # Main reconstruction script
└── README.md                 # Documentation (this file)
```

---

## Setup Instructions

### 1 Prerequisites

Ensure you have **Python ≥ 3.8** and **CUDA** (if using GPU reconstruction).

### 2 Install Dependencies

Install all required Python packages:
```bash
pip install numpy astra-toolbox tifffile matplotlib
```

If you are on Linux and want CUDA acceleration:
```bash
pip install astra-toolbox[cuda]
```

---

## Input Data Preparation

- Place your projection images in the folder:
  ```
  ./raw/projections/
  ```
- Each projection should be a `.tif` image of identical dimensions.
- Example: `proj_0001.tif`, `proj_0002.tif`, ..., `proj_0360.tif`

#### source: "https://drive.google.com/drive/folders/1eJ27DzfssGl5s2eUk7Hzn5H27Xl8iikA"

---

## Running the Reconstruction

You can directly run the script:
```bash
python 3d_reconstruct.py
```

This will:
1. Load all `.tif` projection images from `./raw/projections/`
2. Apply negative log transform preprocessing
3. Reconstruct each slice sequentially using **SART_CUDA**
4. Save all output slices into:
   ```
   ./raw/reconstruction_1/sart_slices_2d/
   ```

---

## Configuration (Inside `3d_reconstruct.py`)

Modify the `GEOMETRY` dictionary to fit your setup:

```python
GEOMETRY = {
    "num_projections": 360,
    "detector_rows": 1000,
    "detector_cols": 1000,
    "sod_mm": 160.0,
    "sdd_mm": 200.0,
    "pixel_size_mm": 0.048,
    "cor_pixel": 518.0,
    "v_center_pixel": 500.0,
    "recon_z_min": 0,
    "recon_z_max": 999,
    "recon_xy_dim": 1000,
    "num_iterations": 50,
    "voxel_size_mm": 0.0384,
    "algorithm": "SIRT_CUDA"
}
```

> You can change the algorithm to `"SART_CUDA"` or `"SIRT_CUDA"` depending on reconstruction preference.

---

## Output

Each reconstructed slice will be saved as:
```
./raw/reconstruction_1/sart_slices_2d/slice_0000.tif
./raw/reconstruction_1/sart_slices_2d/slice_0001.tif
...
```

**Note:-** The output will to lettle to dark to see directly and see structure beacuse of low pixal value of image in 'tif' file. We will increase brightness of image.

![image](./3D_CT_Reconstruction_python/images/Screenshot 2025-10-30 223322.png)


---

## Algorithm Overview

The **SART (Simultaneous Algebraic Reconstruction Technique)** iteratively refines voxel intensities to minimize projection error.  
In this implementation:
- The 3D volume is reconstructed **slice-by-slice**.
- Each slice is reconstructed using a **2D fan-beam** geometry.
- The reconstruction leverages the **ASTRA CUDA backend** for high performance.

---


## Author
**Gaurav Kumar**
Email: *gaurav_k@mfs.iitr.ac.in*  