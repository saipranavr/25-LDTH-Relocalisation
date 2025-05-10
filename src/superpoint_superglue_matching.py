import json
import torch
import cv2
import numpy as np
from PIL import Image
import math
import sys
import os

# Project-specific imports
# Assuming this script (superpoint_superglue_matching.py) is in src/
# and evaluate_estimations.py (the typical entry point) is at the project root.
# evaluate_estimations.py adds the project root to sys.path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.utility.get_satellite_image import get_satellite_image

# --- Constants and Configurations ---
SAT_IMG_SIZE_PX = 256  # Satellite images are fetched at this resolution (width, height)
SEARCH_GRID_POINTS = 10 # Increased grid density
# SEARCH_RADIUS_DEG is no longer used as search is over GeoJSON extent
ROTATION_ANGLES = [0, 90, 180, 270] # Angles to try for satellite image rotation
SCALE_FACTORS = [3.0, 4.0, 5.0] # Adjusted for potentially higher-res drone images vs sat GSD

# --- Model Configuration ---
DEFAULT_SUPERPOINT_CONFIG = {
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
    'max_keypoints': 1024
}

DEFAULT_SUPERGLUE_CONFIG = {
    'weights': 'outdoor', # or 'indoor'
    'sinkhorn_iterations': 20,
    'match_threshold': 0.2,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Model Loading ---
# Define path to the SuperGluePretrainedNetwork repository, which contains both model codes and weights
MODELS_BASE_PATH = os.path.join(PROJECT_ROOT, 'third_party/SuperGluePretrainedNetwork')

SUPERPOINT_MODEL_PATH_IN_SG_REPO = os.path.join(MODELS_BASE_PATH, 'models/superpoint.py') # Path to SuperPoint model definition
SUPERGLUE_MODEL_PATH_IN_SG_REPO = os.path.join(MODELS_BASE_PATH, 'models/superglue.py') # Path to SuperGlue model definition
SUPERPOINT_WEIGHTS_PATH_IN_SG_REPO = os.path.join(MODELS_BASE_PATH, 'models/weights/superpoint_v1.pth')
# SuperGlue weights (e.g., superglue_outdoor.pth) are loaded by the SuperGlue class itself from models/weights/

superpoint_model = None
superglue_model = None

# --- Model Loading (Direct Import from SuperGluePretrainedNetwork for both models) ---
try:
    print(f"Attempting to load SuperPoint model definition from: {MODELS_BASE_PATH}")
    if not os.path.isdir(MODELS_BASE_PATH): # Check if the base directory for models exists
        raise FileNotFoundError(f"Models base directory not found at {MODELS_BASE_PATH}")
    
    # Add the 'models' subdirectory of SuperGluePretrainedNetwork to sys.path to import SuperPoint and SuperGlue
    models_subdir_path = os.path.join(MODELS_BASE_PATH, 'models')
    if not os.path.isdir(models_subdir_path):
        raise FileNotFoundError(f"Models subdirectory not found at {models_subdir_path}")
        
    if models_subdir_path not in sys.path:
        sys.path.insert(0, models_subdir_path) # Add '.../SuperGluePretrainedNetwork/models' to path

    # Import SuperPoint model class
    # The file is models/superpoint.py, so the module is 'superpoint'
    from superpoint import SuperPoint as SP_Model_from_SG_repo
    
    superpoint_model = SP_Model_from_SG_repo(DEFAULT_SUPERPOINT_CONFIG).eval().to(DEVICE)
    
    if not os.path.isfile(SUPERPOINT_WEIGHTS_PATH_IN_SG_REPO):
        raise FileNotFoundError(f"SuperPoint weights not found at {SUPERPOINT_WEIGHTS_PATH_IN_SG_REPO}")
    superpoint_model.load_state_dict(torch.load(SUPERPOINT_WEIGHTS_PATH_IN_SG_REPO, map_location=lambda storage, loc: storage.cuda(DEVICE) if DEVICE.type == 'cuda' else storage.cpu()))
    print("SuperPoint model loaded and weights assigned successfully (using code and weights from SuperGluePretrainedNetwork).")

except Exception as e:
    print(f"Error loading SuperPoint model (from SuperGluePretrainedNetwork): {e}")
    superpoint_model = None
finally:
    # Clean up sys.path if we added models_subdir_path
    if models_subdir_path in sys.path and sys.path[0] == models_subdir_path:
        sys.path.pop(0)

try:
    print(f"Attempting to load SuperGlue model definition from: {MODELS_BASE_PATH}")
    if not os.path.isdir(MODELS_BASE_PATH):
        raise FileNotFoundError(f"Models base directory not found at {MODELS_BASE_PATH}")

    models_subdir_path = os.path.join(MODELS_BASE_PATH, 'models')
    if not os.path.isdir(models_subdir_path):
        raise FileNotFoundError(f"Models subdirectory not found at {models_subdir_path}")

    if models_subdir_path not in sys.path:
        sys.path.insert(0, models_subdir_path) # Add '.../SuperGluePretrainedNetwork/models' to path

    # Import SuperGlue model class
    # The file is models/superglue.py, so the module is 'superglue'
    from superglue import SuperGlue as SG_Model_from_SG_repo
    
    # SuperGlue model from the repo loads its own weights based on config (e.g., 'outdoor')
    # It expects weights like 'superglue_outdoor.pth' to be in its 'weights' subdirectory.
    # The DEFAULT_SUPERGLUE_CONFIG already has 'weights': 'outdoor'.
    # The SuperGlue class in the repo correctly forms the path to 'models/weights/superglue_outdoor.pth'.
    current_sg_config = DEFAULT_SUPERGLUE_CONFIG.copy()
    # Ensure the weights string doesn't include '.pth' if the class adds it.
    # The class seems to expect 'outdoor' or 'indoor', not the full filename.
    # current_sg_config['weights'] = 'superglue_outdoor' # if it needs full name
    
    superglue_model = SG_Model_from_SG_repo(current_sg_config).eval().to(DEVICE)
    print("SuperGlue model loaded successfully (using code and weights from SuperGluePretrainedNetwork).")

except Exception as e:
    print(f"Error loading SuperGlue model (from SuperGluePretrainedNetwork): {e}")
    superglue_model = None
finally:
    # Clean up sys.path if we added models_subdir_path
    if models_subdir_path in sys.path and sys.path[0] == models_subdir_path:
        sys.path.pop(0)


def load_image(image_path: str, grayscale: bool = True):
    """Loads an image from a path and converts to grayscale if specified."""
    img = Image.open(image_path)
    if grayscale:
        img = img.convert('L')
    return np.array(img)

def preprocess_image_to_tensor(image_np: np.ndarray):
    """Converts a NumPy image to a PyTorch tensor."""
    # Normalize image (example: 0-255 to 0-1)
    image_np = image_np.astype(np.float32) / 255.0
    # Add batch and channel dimensions if grayscale [H, W] -> [B, C, H, W]
    if image_np.ndim == 2:
        image_np = image_np[None, None, :, :] # B=1, C=1, H, W
    elif image_np.ndim == 3: # Assuming [H, W, C] for color, but we usually convert to L first
        image_np = np.transpose(image_np, (2, 0, 1)) # To [C, H, W]
        image_np = image_np[None, :, :, :] # To [B, C, H, W]
    else:
        raise ValueError(f"Unsupported image dimensions: {image_np.ndim}")
    return torch.from_numpy(image_np).to(DEVICE)

def detect_features_superpoint(image_tensor: torch.Tensor, model: torch.nn.Module):
    """Detects features using SuperPoint."""
    if model is None:
        print("SuperPoint model not loaded. Skipping feature detection.")
        return torch.empty(0, 2, device=DEVICE), torch.empty(0, 256, device=DEVICE), torch.empty(0, device=DEVICE)
    
    with torch.no_grad():
        # SuperPoint typically expects a batch of grayscale images: (B, 1, H, W)
        # Ensure image_tensor is on the correct device
        pred = model({'image': image_tensor.to(DEVICE)})

    # Extract keypoints, descriptors, and scores
    # The exact keys might vary slightly based on the SuperPoint implementation
    # Common keys: 'keypoints', 'descriptors', 'scores'
    # Keypoints are typically [B, N, 2] (x, y)
    # Descriptors are typically [B, N, D] (D is descriptor dim, e.g., 256)
    # Scores are typically [B, N]
    keypoints = pred.get('keypoints', [torch.empty(0,2, device=DEVICE)])[0] # Get first batch item
    descriptors = pred.get('descriptors', [torch.empty(0,256, device=DEVICE)])[0]
    scores = pred.get('scores', [torch.empty(0, device=DEVICE)])[0]
    
    # The SuperPoint model from SuperGluePretrainedNetwork/models/superpoint.py
    # already returns descriptors in the shape [D, N] (DescriptorDimension, NumKeypoints)
    # as an item in a list.
    # pred.get('descriptors')[0] gives us this [D, N] tensor.
    # This is the format SuperGlue expects for its input (after adding a batch dimension).
    # So, the previous transpose logic is no longer needed and was causing the error.
    
    # Ensure descriptors has the expected shape [D, N]
    # If N=0, shape might be [D,0] or [0,D] or just [0]. If N>0, D should be 256.
    if descriptors.numel() > 0 and descriptors.shape[0] != DEFAULT_SUPERPOINT_CONFIG.get('descriptor_dim', 256):
        # If descriptor_dim is not in config, assume 256.
        # This check is a safeguard. If this happens, something is unexpected.
        print(f"Warning: Descriptor dimension mismatch. Expected {DEFAULT_SUPERPOINT_CONFIG.get('descriptor_dim', 256)}, Got {descriptors.shape[0]}. Descriptors shape: {descriptors.shape}")

    print(f"SuperPoint detected {keypoints.shape[0]} keypoints.")
    return keypoints, descriptors, scores


def match_features_superglue(kpts0: torch.Tensor, desc0: torch.Tensor, scores0: torch.Tensor, image0_tensor: torch.Tensor,
                             kpts1: torch.Tensor, desc1: torch.Tensor, scores1: torch.Tensor, image1_tensor: torch.Tensor,
                             model: torch.nn.Module):
    """
    Matches features using SuperGlue.
    Assumes kpts are [N, 2], desc are [D, N], scores are [N].
    Image tensors are [B, C, H, W] (B=1, C=1 for grayscale).
    The SuperGlue model in third_party/SuperGluePretrainedNetwork/models/superglue.py expects 'image0' and 'image1'.
    """
    if model is None:
        print("SuperGlue model not loaded. Skipping feature matching.")
        # Return structure expected by relocalize_drone: matched_indices0, matched_indices1, confidence
        return torch.empty(0, device=DEVICE, dtype=torch.long), \
               torch.empty(0, device=DEVICE, dtype=torch.long), \
               torch.empty(0, device=DEVICE)

    # Prepare input for SuperGlue
    # Keypoints are [N, 2] (x,y format from SuperPoint)
    # Descriptors are [D, N]
    # Scores are [N]
    # Image tensors are [1, 1, H, W]
    pred_input = {
        'keypoints0': kpts0[None].to(DEVICE),    # Batch dim: [1, N0, 2]
        'descriptors0': desc0[None].to(DEVICE),  # Batch dim: [1, D, N0]
        'scores0': scores0[None].to(DEVICE),     # Batch dim: [1, N0]
        'image0': image0_tensor.to(DEVICE),      # Image tensor [1, 1, H0, W0]

        'keypoints1': kpts1[None].to(DEVICE),    # Batch dim: [1, N1, 2]
        'descriptors1': desc1[None].to(DEVICE),  # Batch dim: [1, D, N1]
        'scores1': scores1[None].to(DEVICE),     # Batch dim: [1, N1]
        'image1': image1_tensor.to(DEVICE)       # Image tensor [1, 1, H1, W1]
    }
    # The SuperGlue model's normalize_keypoints function uses image0.shape and image1.shape directly.
    # No need to pass 'image0_shape_wh' or 'image1_shape_wh' if image tensors are provided.

    with torch.no_grad():
        pred = model(pred_input)

    # `matches0` are indices of kpts1 that match kpts0. Shape [N0]
    # A value of -1 indicates no match.
    # `match_confidence` (or `matching_scores0`) are the confidences. Shape [N0]
    matches = pred.get('matches0', [torch.empty(0, device=DEVICE, dtype=torch.long)])[0]
    confidence = pred.get('matching_scores0', [torch.empty(0, device=DEVICE)])[0]
    
    # Filter out invalid matches (where index is -1)
    # `matches` from SuperGlue are indices of kpts1 that match kpts0. Shape [N0]
    # A value of -1 indicates no match.
    valid_matches_mask = matches > -1
    
    # Indices of keypoints in image0 that have a valid match
    matched_kpts0_indices = torch.arange(len(matches), device=DEVICE)[valid_matches_mask]
    # Indices of corresponding keypoints in image1
    matched_kpts1_indices = matches[valid_matches_mask]
    
    # Confidence scores for the valid matches
    match_confidence = confidence[valid_matches_mask]
    
    print(f"SuperGlue found {len(matched_kpts0_indices)} matches.")
    
    return matched_kpts0_indices, matched_kpts1_indices, match_confidence


def transform_satellite_crop(sat_crop_np: np.ndarray, scale_factor: float, rotation_angle: float):
    """
    Applies scaling and rotation to a fetched satellite crop (NumPy array).
    """
    if sat_crop_np is None:
        return None
        
    sat_img_pil = Image.fromarray(sat_crop_np)

    # 1. Rotate
    # expand=True ensures the whole rotated image is visible.
    # The background color can be set if needed, default is black.
    sat_img_rotated_pil = sat_img_pil.rotate(rotation_angle, resample=Image.Resampling.BICUBIC, expand=True)
    # print(f"Rotated satellite crop by {rotation_angle} degrees.")

    # 2. Scale
    # The scale_factor should represent how much to scale the satellite image
    # to match the drone image's ground sampling distance (GSD) or field of view.
    new_width = int(sat_img_rotated_pil.width * scale_factor)
    new_height = int(sat_img_rotated_pil.height * scale_factor)
    if new_width <= 0 or new_height <=0:
        # print(f"Warning: Invalid new dimensions after scaling ({new_width}x{new_height}). Skipping this transform.")
        return None
    sat_img_scaled_pil = sat_img_rotated_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
    # print(f"Scaled satellite crop by factor {scale_factor} to {new_width}x{new_height}.")

    # Convert to grayscale numpy array if it's not already (get_satellite_image might return RGB)
    if sat_img_scaled_pil.mode == 'RGB' or sat_img_scaled_pil.mode == 'RGBA':
        sat_img_scaled_pil = sat_img_scaled_pil.convert('L')
        
    return np.array(sat_img_scaled_pil)


def calculate_geo_bounding_box(center_lon: float, center_lat: float, size_deg: float):
    """
    Calculates a square geographic bounding box centered at (center_lon, center_lat)
    with a given size in degrees.
    Note: This is a simplification. A fixed degree size has different metric dimensions
    at different latitudes. For more precision, convert metric size to degrees based on latitude.
    """
    half_size_deg = size_deg / 2.0
    min_lon = center_lon - half_size_deg
    max_lon = center_lon + half_size_deg
    min_lat = center_lat - half_size_deg
    max_lat = center_lat + half_size_deg
    return min_lon, min_lat, max_lon, max_lat


def parse_geojson_bounds(geojson_path: str):
    """Parses a GeoJSON file to extract the overall bounding box."""
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Assuming the GeoJSON structure from rickmansworth_example_search_area.json
        # It's a FeatureCollection with one Polygon feature.
        coords = geojson_data['features'][0]['geometry']['coordinates'][0]
        
        # Coords is a list of [lon, lat] pairs
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
        
        return min(lons), min(lats), max(lons), max(lats)
    except Exception as e:
        print(f"Error parsing GeoJSON {geojson_path}: {e}")
        return None, None, None, None

def relocalize_drone(
    drone_image_path: str,
    # initial_est_lon: float, # No longer using a single point initial estimate for search radius
    # initial_est_lat: float,
    search_area_geojson_path: str, 
    grid_points: int = SEARCH_GRID_POINTS, # Number of points per side of the grid
    rotation_angles: list = ROTATION_ANGLES,
    scale_factors: list = SCALE_FACTORS,
    satellite_crop_size_deg: float = 0.002 # Adjusted based on debug visualization (approx 222m across 1024px)
):
    """
    Main relocalization pipeline.
    Scans satellite crops across a search area defined by GeoJSON, matches features, 
    and returns the best geo-location.
    """
    min_search_lon, min_search_lat, max_search_lon, max_search_lat = parse_geojson_bounds(search_area_geojson_path)
    if min_search_lon is None: # Parsing failed
        print(f"Could not determine search bounds from {search_area_geojson_path}. Aborting relocalization.")
        # Fallback: return a clearly invalid coordinate or raise error
        return -999.0, -999.0 

    if superpoint_model is None or superglue_model is None:
        print("SuperPoint or SuperGlue model not loaded. Cannot proceed with relocalization.")
        return -999.0, -999.0 

    # Load drone image once
    drone_img_np = load_image(drone_image_path, grayscale=True) # H, W
    if drone_img_np is None:
        print(f"Failed to load drone image: {drone_image_path}")
        return -999.0, -999.0
    drone_img_tensor = preprocess_image_to_tensor(drone_img_np) # B, C, H, W
    drone_h, drone_w = drone_img_np.shape[:2]

    print(f"\n--- Detecting features for Drone Image: {drone_image_path} ---")
    kpts_drone_raw, desc_drone_raw, scores_drone_raw = detect_features_superpoint(drone_img_tensor, superpoint_model)
    if kpts_drone_raw.numel() == 0:
        print("No keypoints detected in drone image. Cannot proceed.")
        return -999.0, -999.0

    best_match_info = {
        "score": -1.0,
        "lon": -999.0, # Use clearly invalid initial lon/lat
        "lat": -999.0,
        "scale": 1.0,
        "rotation": 0.0,
        "num_inliers": 0,
        # Store matched keypoints for the best match for potential refinement later
        "mkpts_drone": None, 
        "mkpts_sat": None,
        "best_sat_crop_transformed_np": None, # Store the best transformed crop for visualization/refinement
        "best_sat_crop_original_w": SAT_IMG_SIZE_PX, # Assuming fixed fetch size for now
        "best_sat_crop_original_h": SAT_IMG_SIZE_PX,
        "best_sat_crop_min_lon": None, # Geo bounds of the best original satellite crop
        "best_sat_crop_min_lat": None,
        "best_sat_crop_max_lon": None,
        "best_sat_crop_max_lat": None,
    }

    # Create a search grid spanning the entire GeoJSON area
    # If grid_points is 1, use the center of the search area.
    if grid_points > 1:
        lon_coords = np.linspace(min_search_lon, max_search_lon, grid_points)
        lat_coords = np.linspace(min_search_lat, max_search_lat, grid_points)
    else: # grid_points == 1 or invalid
        lon_coords = np.array([(min_search_lon + max_search_lon) / 2])
        lat_coords = np.array([(min_search_lat + max_search_lat) / 2])


    total_searches = len(lon_coords) * len(lat_coords) * len(rotation_angles) * len(scale_factors)
    current_search_count = 0
    print(f"Starting search across {total_searches} satellite crop variations...")

    for R_idx, R_val in enumerate(rotation_angles):
        for S_idx, S_val in enumerate(scale_factors):
            for lon_idx, center_lon in enumerate(lon_coords):
                for lat_idx, center_lat in enumerate(lat_coords):
                    current_search_count += 1
                    print(f"\n--- Attempt {current_search_count}/{total_searches}: Sat crop centered near ({center_lon:.6f}, {center_lat:.6f}), Rot: {R_val}, Scale: {S_val} ---")

                    # 1. Get satellite crop for this geo-coordinate center
                    # The bounding box for get_satellite_image defines the crop.
                    crop_min_lon, crop_min_lat, crop_max_lon, crop_max_lat = calculate_geo_bounding_box(center_lon, center_lat, satellite_crop_size_deg)
                    
                    raw_sat_crop_original_np = get_satellite_image(crop_min_lon, crop_min_lat, crop_max_lon, crop_max_lat)

                    if raw_sat_crop_original_np is None:
                        continue
                    
                    # 2. Transform this satellite crop (rotate, scale)
                    sat_crop_transformed_np = transform_satellite_crop(raw_sat_crop_original_np, S_val, R_val)
                    if sat_crop_transformed_np is None:
                        # print("Satellite crop transformation failed.")
                        continue

                    sat_img_tensor = preprocess_image_to_tensor(sat_crop_transformed_np)
                    sat_h, sat_w = sat_crop_transformed_np.shape[:2]
                    if sat_h == 0 or sat_w == 0:
                        # print("Transformed satellite crop has zero dimension. Skipping.")
                        continue
                    
                    # print("Detecting features for transformed satellite crop...")
                    kpts_sat_raw, desc_sat_raw, scores_sat_raw = detect_features_superpoint(sat_img_tensor, superpoint_model)
                    if kpts_sat_raw.numel() == 0:
                        # print("No keypoints detected in this satellite crop. Skipping.")
                        continue

                    # print("Matching features with SuperGlue...")
                    # Pass the image tensors (drone_img_tensor, sat_img_tensor) to match_features_superglue
                    matched_indices_drone, matched_indices_sat, match_confidence = match_features_superglue(
                        kpts_drone_raw, desc_drone_raw, scores_drone_raw, drone_img_tensor,
                        kpts_sat_raw, desc_sat_raw, scores_sat_raw, sat_img_tensor,
                        superglue_model
                    )
                    
                    mkpts_drone_tensor = kpts_drone_raw[matched_indices_drone]
                    mkpts_sat_tensor = kpts_sat_raw[matched_indices_sat]

                    # --- Geometric Verification using Homography (RANSAC) ---
                    current_ransac_inliers = 0
                    if len(matched_indices_drone) >= 4: # Need at least 4 points for homography
                        H, ransac_mask = cv2.findHomography(
                            mkpts_drone_tensor.cpu().numpy(), 
                            mkpts_sat_tensor.cpu().numpy(), 
                            cv2.RANSAC, 
                            ransacReprojThreshold=5.0 # This threshold might need tuning
                        )
                        if H is not None:
                            current_ransac_inliers = np.sum(ransac_mask)
                        # print(f"RANSAC inliers: {current_ransac_inliers} out of {len(matched_indices_drone)} SuperGlue matches.")
                    
                    # Use number of RANSAC inliers as the score
                    current_score = current_ransac_inliers

                    if current_score > best_match_info["score"]:
                        best_match_info["score"] = current_score
                        best_match_info["lon"] = center_lon # Geo-center of the satellite tile
                        best_match_info["lat"] = center_lat
                        best_match_info["scale"] = S_val
                        best_match_info["rotation"] = R_val
                        best_match_info["num_inliers"] = current_ransac_inliers # Store RANSAC inliers
                        best_match_info["mkpts_drone"] = mkpts_drone_tensor
                        best_match_info["mkpts_sat"] = mkpts_sat_tensor
                        best_match_info["best_sat_crop_transformed_np"] = sat_crop_transformed_np
                        best_match_info["best_sat_crop_original_w"] = raw_sat_crop_original_np.shape[1]
                        best_match_info["best_sat_crop_original_h"] = raw_sat_crop_original_np.shape[0]
                        best_match_info["best_sat_crop_min_lon"] = crop_min_lon
                        best_match_info["best_sat_crop_min_lat"] = crop_min_lat
                        best_match_info["best_sat_crop_max_lon"] = crop_max_lon
                        best_match_info["best_sat_crop_max_lat"] = crop_max_lat
                        
                        print(f"✨ New best match: {current_ransac_inliers} RANSAC inliers at ({center_lon:.6f}, {center_lat:.6f}), Scale: {S_val}, Rot: {R_val}")

    if best_match_info["num_inliers"] > 0: # Check if any valid match was found
        print(f"\n🏆 Best overall match: {best_match_info['num_inliers']} RANSAC inliers, Score: {best_match_info['score']:.2f}")
        print(f"  Tile Center (Lon, Lat): ({best_match_info['lon']:.6f}, {best_match_info['lat']:.6f})")
        print(f"  Transform: Scale {best_match_info['scale']}, Rotation {best_match_info['rotation']}")
        
        # TODO: Refine location using homography and best_match_info['mkpts_drone'], ['mkpts_sat'], etc.
        # For now, returning the center of the best satellite tile.
        final_lon, final_lat = best_match_info["lon"], best_match_info["lat"]
        return final_lon, final_lat
    else:
        print("\n⚠️ No satisfactory match found across the search space.")
        return -999.0, -999.0 # Return clearly invalid if no match


def main_matching_pipeline(drone_image_path: str, satellite_image_path: str): # This function is now largely superseded by relocalize_drone
    """
    Original main pipeline for loading images, detecting, and matching features.
    Kept for reference, but relocalize_drone is the new entry point.
    """
    # TODO: Determine how to get/estimate these parameters
    estimated_sat_center = (500, 500) # Placeholder: (x,y) center in satellite image for cropping
    initial_rotation_angle = 0.0 # Placeholder: degrees
    initial_scale_factor = 1.0 # Placeholder: factor

    # Load drone image
    drone_img_np = load_image(drone_image_path, grayscale=True) # H, W
    drone_img_tensor = preprocess_image_to_tensor(drone_img_np) # B, C, H, W
    drone_h, drone_w = drone_img_np.shape[:2]

    # This part needs to be re-thought as we fetch satellite crops iteratively
    print(f"Legacy: Processing satellite image: {satellite_image_path}")
    # The old preprocess_satellite_image took a path and did cropping.
    # The new transform_satellite_crop takes a numpy array.
    # For a single image test, one might do:
    raw_sat_img = load_image(satellite_image_path, grayscale=False) # Load as is, could be color
    sat_img_processed_np = transform_satellite_crop(raw_sat_img, initial_scale_factor, initial_rotation_angle)
    
    if sat_img_processed_np is None:
        print("Failed to process satellite image in legacy pipeline.")
        return None, None, None

    sat_img_tensor = preprocess_image_to_tensor(sat_img_processed_np) # B, C, H, W
    sat_h, sat_w = sat_img_processed_np.shape[:2]


    # --- Feature Detection (SuperPoint) ---
    print("\n--- SuperPoint Feature Detection ---")
    if superpoint_model is not None:
        kpts_drone_raw, desc_drone_raw, scores_drone_raw = detect_features_superpoint(drone_img_tensor, superpoint_model)
        kpts_sat_raw, desc_sat_raw, scores_sat_raw = detect_features_superpoint(sat_img_tensor, superpoint_model)
    else:
        print("SuperPoint model not available, using placeholder data.")
        kpts_drone_raw, desc_drone_raw, scores_drone_raw = torch.rand(100, 2, device=DEVICE), torch.rand(256, 100, device=DEVICE), torch.rand(100, device=DEVICE)
        kpts_sat_raw, desc_sat_raw, scores_sat_raw = torch.rand(120, 2, device=DEVICE), torch.rand(256, 120, device=DEVICE), torch.rand(120, device=DEVICE)


    # --- Feature Matching (SuperGlue) ---
    print("\n--- SuperGlue Feature Matching ---")
    if superglue_model is not None and kpts_drone_raw.numel() > 0 and kpts_sat_raw.numel() > 0:
        # SuperGlue needs original image shapes (W,H) for normalization of keypoints
        drone_shape_wh = (drone_w, drone_h)
        sat_shape_wh = (sat_w, sat_h)
        
        matched_indices_drone, matched_indices_sat, match_confidence = match_features_superglue(
            kpts_drone_raw, desc_drone_raw, scores_drone_raw,
            kpts_sat_raw, desc_sat_raw, scores_sat_raw,
            drone_shape_wh, sat_shape_wh,
            superglue_model
        )
        
        # Get the actual matched keypoints
        mkpts_drone = kpts_drone_raw[matched_indices_drone]
        mkpts_sat = kpts_sat_raw[matched_indices_sat]

    else:
        print("SuperGlue model not available or no keypoints detected, using placeholder matches.")
        num_placeholder_matches = min(kpts_drone_raw.shape[0], kpts_sat_raw.shape[0], 10)
        mkpts_drone = kpts_drone_raw[:num_placeholder_matches]
        mkpts_sat = kpts_sat_raw[:num_placeholder_matches] # Simplistic pairing
        match_confidence = torch.rand(num_placeholder_matches, device=DEVICE)


    # Further steps:
    # 1. Filter matches based on confidence (already done by SuperGlue's threshold or can be done again).
    # 2. Use matches to estimate transformation (e.g., homography using cv2.findHomography).
    # 3. Visualize matches (e.g., using cv2.drawMatches).

    if mkpts_drone.numel() > 0 and mkpts_sat.numel() > 0:
        print(f"\nSuccessfully matched {mkpts_drone.shape[0]} keypoints.")
        # Example: visualize_matches(drone_img_np, mkpts_drone, sat_img_processed_np, mkpts_sat, match_confidence)
    else:
        print("Matching failed or no keypoints to match.")

    return mkpts_drone, mkpts_sat, match_confidence


if __name__ == '__main__':
    # Example usage:
    example_drone_image_path = "data/example/images/rickmansworth_example.jpg"
    
    # Initial estimate for Rickmansworth example (from data/example/estimations.csv)
    initial_lon = -0.523226
    initial_lat = 51.681212
    
    # Path to the GeoJSON defining the broader search area (though not directly used by get_satellite_image for single fetch)
    # search_area_geojson = "data/example/images/rickmansworth_example_search_area.json"

    print(f"Starting relocalization for {example_drone_image_path}")
    print(f"Initial estimated location: Lon={initial_lon}, Lat={initial_lat}")

    # Check if drone image exists
    try:
        # Test if models loaded
        if superpoint_model is None or superglue_model is None:
            raise RuntimeError("SuperPoint or SuperGlue models failed to load. Aborting example.")

        with open(example_drone_image_path, 'rb') as f: # Check file existence
            pass 
        print(f"Found example drone image: {example_drone_image_path}")

        # Run the new relocalization pipeline
        final_lon, final_lat = relocalize_drone(
            drone_image_path=example_drone_image_path,
            initial_est_lon=initial_lon,
            initial_est_lat=initial_lat
            # search_area_geojson_path=search_area_geojson # Optional for now
        )
        
        print(f"\n--- Relocalization Complete ---")
        print(f"Initial Estimated Coords: (Lon: {initial_lon:.6f}, Lat: {initial_lat:.6f})")
        print(f"Final Estimated Coords:   (Lon: {final_lon:.6f}, Lat: {final_lat:.6f})")

    except FileNotFoundError:
        print(f"Error: Example drone image not found at {example_drone_image_path}")
        print("Please update the path or provide appropriate example images.")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the example run: {e}")


    print("\nScript execution finished.")
