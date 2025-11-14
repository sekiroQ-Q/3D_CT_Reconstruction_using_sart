import numpy as np
import astra
import tifffile
import glob
import os

# --- 1. Configuration ---

PROJ_FOLDER = './raw/projections'
RECON_FOLDER = './raw/reconstruction_1'

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
    "algorithm": "SIRT_CUDA"  # 2D algorithm
}

# 2. Load Projections

def load_projections(path, num_projections, rows, cols):
    """Load projection images into 3D array"""
    print(f"Loading projections from: {path}")

    #collect all the .tif images using glob
    file_list = sorted(glob.glob(os.path.join(path, '*.tif')))

    #we check for errors: maybe the folder has no files, or the number of files doesn’t match
    if len(file_list) == 0:
        print(f"Error: No .tif files found in '{path}'.")
        return None
    
    if len(file_list) != num_projections:
        print(f"Warning: Found {len(file_list)} files, expected {num_projections}.")
        num_projections = len(file_list)

    #create a NumPy array to store all projection images
    projections_data = np.zeros((rows, num_projections, cols), dtype=np.float32)

    #We now iterate through each projection file
    for i, f_path in enumerate(file_list):
        if (i % 50 == 0):
            print(f"  Loading file {i+1} of {num_projections}...")
        #We load each .tif image, check that its size matches the detector dimensions
        img = tifffile.imread(f_path).astype(np.float32)
        if img.shape != (rows, cols):
            print(f"Error: Image {f_path} has wrong shape {img.shape}. Expected {(rows, cols)}.")
            return None
            
        projections_data[:, i, :] = img

    print(f"Successfully loaded {num_projections} projections.")
    print(f"  Raw projection data range: [{np.min(projections_data):.2f}, {np.max(projections_data):.2f}]")

    # Preprocessing: Negative-log transform 
    print("\nApplying negative-log preprocessing...")
    
    i_zero = np.max(projections_data)
    i_min = np.min(projections_data)
    
    print(f"  Raw data range: [{i_min:.2f}, {i_zero:.2f}]")
    
    if i_zero < 10.0:
        print("  WARNING: Data appears already preprocessed. Skipping log transform.")
        return projections_data
    
    # Standard log transform
    projections_data = projections_data / i_zero
    projections_data = np.clip(projections_data, 1e-6, 1.0)
    projections_data = -np.log(projections_data)
    
    print(f"  After log transform: [{np.min(projections_data):.4f}, {np.max(projections_data):.4f}]")

    return projections_data

# 3. Reconstruct Single Slice (2D SART) 

def reconstruct_slice_2d(sinogram, geom_params, slice_idx):
    """
    Reconstruct a single 2D slice using SART.
    sinogram: 2D array of shape (num_projections, detector_cols)
    """
    
    # Get geometry parameters
    sod_mm = geom_params["sod_mm"]
    sdd_mm = geom_params["sdd_mm"]
    pixel_size_mm = geom_params["pixel_size_mm"]
    cor_pixel = geom_params["cor_pixel"]
    det_cols = geom_params["detector_cols"]
    num_projections = geom_params["num_projections"]
    recon_size = geom_params["recon_xy_dim"]
    voxel_size = geom_params["voxel_size_mm"]
    num_iterations = geom_params["num_iterations"]
    
    # Convert to voxel units
    sod_voxels = sod_mm / voxel_size
    sdd_voxels = sdd_mm / voxel_size
    pixel_size_voxels = pixel_size_mm / voxel_size
    d_origin_det_voxels = sdd_voxels - sod_voxels
    det_x_offset_pix = cor_pixel - (det_cols / 2.0)
    det_x_offset_voxels = det_x_offset_pix * pixel_size_voxels
    
    # Create 2D volume geometry
    vol_geom = astra.create_vol_geom(recon_size, recon_size)
    
    # Create 2D projection geometry (fan-beam)
    angles = np.linspace(0, 2 * np.pi, num_projections, endpoint=False)
    
    proj_geom = astra.create_proj_geom(
        'fanflat',
        pixel_size_voxels,
        det_cols,
        angles,
        sod_voxels,
        d_origin_det_voxels
    )
    
    # Add detector offset if needed
    if abs(det_x_offset_voxels) > 0.01:
        for i in range(num_projections):
            proj_geom['ProjectionAngles'][i] = angles[i]
            proj_geom['DetectorRowCount'] = 1
            proj_geom['DetectorColCount'] = det_cols
    
    # Create ASTRA data structures
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    recon_id = astra.data2d.create('-vol', vol_geom)
    
    # Configure SART algorithm
    cfg = astra.astra_dict(geom_params["algorithm"])
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    
    # Create and run algorithm
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iterations)
    
    # Get result
    recon_slice = astra.data2d.get(recon_id)
    
    # Cleanup
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(recon_id)
    astra.data2d.delete(sinogram_id)
    
    return recon_slice

# 4. Main Execution

def main():
    print("="*60)
    print("Starting 2D Slice-by-Slice CT Reconstruction")
    print("="*60)
    
    OUTPUT_SLICE_FOLDER = os.path.join(RECON_FOLDER, 'sart_slices_2d')
    os.makedirs(OUTPUT_SLICE_FOLDER, exist_ok=True)

    # Step 1: Load Projections
    projections_data = load_projections(
        PROJ_FOLDER,
        GEOMETRY["num_projections"],
        GEOMETRY["detector_rows"],
        GEOMETRY["detector_cols"]
    )

    if projections_data is None:
        return

    # projections_data shape: (detector_rows, num_projections, detector_cols)
    
    # Step 2: Reconstruct Slice-by-Slice 
    z_min = GEOMETRY["recon_z_min"]
    z_max = GEOMETRY["recon_z_max"]
    num_slices = z_max - z_min + 1
    
    print(f"\n{'='*60}")
    print(f"Reconstructing {num_slices} slices (from {z_min} to {z_max})")
    print(f"Algorithm: {GEOMETRY['algorithm']}")
    print(f"Iterations per slice: {GEOMETRY['num_iterations']}")
    print(f"{'='*60}\n")
    
    for slice_idx in range(z_min, z_max + 1):
        # Extract sinogram for this slice
        # sinogram shape should be: (num_projections, detector_cols)
        # projections_data is (detector_rows, num_projections, detector_cols)
        sinogram = projections_data[slice_idx, :, :]  # Already correct shape
        
        # Reconstruct this slice
        recon_slice = reconstruct_slice_2d(sinogram, GEOMETRY, slice_idx)
        
        # Save the slice
        filename = f"slice_{str(slice_idx).zfill(4)}.tif"
        save_path = os.path.join(OUTPUT_SLICE_FOLDER, filename)
        tifffile.imwrite(save_path, recon_slice.astype(np.float32))
        
        # Progress update
        progress = slice_idx - z_min + 1
        if progress % 50 == 0 or progress == 1 or progress == num_slices:
            print(f"  [{progress}/{num_slices}] Slice {slice_idx} reconstructed: "
                  f"min={np.min(recon_slice):.4f}, max={np.max(recon_slice):.4f}, "
                  f"mean={np.mean(recon_slice):.4f}")
    
    print(f"\n{'='*60}")
    print(f"Reconstruction complete!")
    print(f"{num_slices} slices saved to: {OUTPUT_SLICE_FOLDER}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
