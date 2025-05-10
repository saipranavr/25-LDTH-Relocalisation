import sys
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path to allow imports from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utility.get_satellite_image import get_satellite_image
from src.superpoint_superglue_matching import (
    load_image,
    calculate_geo_bounding_box,
    transform_satellite_crop,
    SCALE_FACTORS # Import the list of scale factors
)

# Define the default satellite crop size for this debug script.
# Let's try a larger area to see if we get better base resolution from the server.
SATELLITE_CROP_SIZE_DEG_DEBUG_DEFAULT = 0.002 # Approx 222m, GSD will be ~0.22m for 1024px image

def load_truth_coords(truth_csv_path: str, image_id: str) -> tuple[float, float] | None:
    """Loads true longitude and latitude for a given image_id from the truth CSV."""
    try:
        with open(truth_csv_path, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['image_id'] == image_id:
                    return float(row['longitude']), float(row['latitude'])
        print(f"Image ID {image_id} not found in {truth_csv_path}")
        return None
    except Exception as e:
        print(f"Error reading truth CSV {truth_csv_path}: {e}")
        return None

def main_visualize_truth(
    image_id: str = "rickmansworth_example",
    data_dir_base: str = "data/example",
    satellite_crop_size_deg: float = SATELLITE_CROP_SIZE_DEG_DEBUG_DEFAULT,
    scale_factor_to_apply: float = 1.0 # Default to 1.0 as get_satellite_image now fetches higher res
):
    """
    Visualizes the drone image alongside a satellite crop fetched at the ground truth coordinates.
    """
    truth_csv_path = os.path.join(PROJECT_ROOT, data_dir_base, "truth.csv")
    drone_image_path = os.path.join(PROJECT_ROOT, data_dir_base, "images", f"{image_id}.jpg")

    true_coords = load_truth_coords(truth_csv_path, image_id)
    if true_coords is None:
        return
    true_lon, true_lat = true_coords
    print(f"Ground truth for {image_id}: Lon={true_lon:.6f}, Lat={true_lat:.6f}")

    # Load drone image
    drone_img_np = load_image(drone_image_path, grayscale=True)
    if drone_img_np is None:
        print(f"Failed to load drone image: {drone_image_path}")
        return
    print(f"Drone image '{drone_image_path}' loaded, shape: {drone_img_np.shape}")

    # Fetch satellite crop at ground truth coordinates
    print(f"Fetching satellite crop at truth coordinates with size_deg={satellite_crop_size_deg}...")
    crop_min_lon, crop_min_lat, crop_max_lon, crop_max_lat = calculate_geo_bounding_box(
        true_lon, true_lat, satellite_crop_size_deg
    )
    sat_crop_original_np = get_satellite_image(crop_min_lon, crop_min_lat, crop_max_lon, crop_max_lat)

    if sat_crop_original_np is None:
        print("Failed to retrieve satellite crop at truth coordinates.")
        return
    print(f"Original satellite crop fetched, shape: {sat_crop_original_np.shape}")
    
    # Apply transformation (scaling, optional rotation for consistency)
    # For this debug view, let's use 0 rotation and a chosen scale factor.
    rotation_to_apply = 0.0 
    print(f"Applying scale={scale_factor_to_apply}, rotation={rotation_to_apply} to satellite crop...")
    sat_crop_transformed_np = transform_satellite_crop(
        sat_crop_original_np,
        scale_factor=scale_factor_to_apply,
        rotation_angle=rotation_to_apply
    )
    if sat_crop_transformed_np is None:
        print("Failed to transform satellite crop.")
        # Fallback to showing original if transform fails but original was fetched
        if sat_crop_original_np is not None:
             sat_crop_transformed_np = sat_crop_original_np # Show original if transform failed
        else:
            return 
            
    print(f"Transformed satellite crop shape: {sat_crop_transformed_np.shape}")

    # Display side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(drone_img_np, cmap='gray')
    axes[0].set_title(f"Drone Image: {image_id}\nShape: {drone_img_np.shape}")
    axes[0].axis('off')

    axes[1].imshow(sat_crop_transformed_np, cmap='gray')
    axes[1].set_title(
        f"Satellite Crop at Truth Coords\n"
        f"(Lon: {true_lon:.4f}, Lat: {true_lat:.4f})\n"
        f"CropSizeDeg: {satellite_crop_size_deg}, AppliedScale: {scale_factor_to_apply}\n"
        f"Shape: {sat_crop_transformed_np.shape}"
    )
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # You can change the image_id or parameters here for different tests
    main_visualize_truth(
        image_id="rickmansworth_example",
        data_dir_base="data/example",
        satellite_crop_size_deg=SATELLITE_CROP_SIZE_DEG_DEBUG_DEFAULT, # Use the new default (0.002)
        scale_factor_to_apply=1.0       # No additional scaling after fetching 1024px image
    )
    # Example for a more zoomed-out satellite view (larger crop_size_deg, less scaling)
    # main_visualize_truth(
    #     image_id="rickmansworth_example",
    #     data_dir_base="data/example",
    #     satellite_crop_size_deg=0.002, 
    #     scale_factor_to_apply=1.0      
    # )
