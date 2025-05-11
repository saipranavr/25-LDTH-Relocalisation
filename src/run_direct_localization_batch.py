import csv
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import argparse
import re
import json # For creating temporary GeoJSONs
import requests # For API calls
from io import BytesIO # For handling image data
from PIL import Image # Added PIL.Image import

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# Define project_root globally for run_command
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

from src.utility.bounding_box import BoundingBox, Projection
from src.utility.epsg_4326_to_3035 import convert_4326_bbox_to_3035_bbox

# Configuration
DEFAULT_SUPERGLUE_SCRIPT_PATH = "third_party/SuperGluePretrainedNetwork/match_pairs.py"
DEFAULT_ESTIMATOR_SCRIPT_PATH = "src/estimate_location_from_matches.py"
# DEFAULT_FETCHER_SCRIPT_PATH is no longer needed as we fetch directly

API_URL = "https://image.discomap.eea.europa.eu/arcgis/rest/services/GioLand/VHR_2018_LAEA/ImageServer/exportImage"
DEFAULT_SUPERGLUE_OUTPUT_DIR = Path("data/example/direct_localization_output/superglue/")
# DEFAULT_TARGET_IMG_WIDTH/HEIGHT will be taken from an argument or inferred if possible.
# For now, let's assume the high-res UAV images are 512x512 as generated previously.
ASSUMED_TARGET_IMG_DIM = 512

# Parameters for refined search
DEFAULT_REFINED_SEARCH_GEO_WIDTH_DEGREES = 0.002  # Approx 200-220m
DEFAULT_REFINED_SEARCH_GEO_HEIGHT_DEGREES = 0.0015 # Approx 160-170m
DEFAULT_REFINED_SEARCH_IMAGE_PIXELS = 1024 # Fetch 1024x1024 image for the refined area

def run_command(command_list):
    """Helper to run shell commands and print their output."""
    print(f"Executing: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=PROJECT_ROOT_PATH) # Ensure CWD
        stdout, _ = process.communicate()
        print(stdout)
        if process.returncode != 0:
            print(f"Error: Command failed with return code {process.returncode}")
            return False, stdout
        return True, stdout
    except Exception as e:
        print(f"Exception running command: {e}")
        return False, str(e)

def parse_estimation_output_for_coords(output_str):
    """Parses the output of estimate_location_from_matches.py to get only lat, lon."""
    est_lat, est_lon = None, None
    lines = output_str.splitlines()
    found_success_header = False
    for line in lines:
        line = line.strip()
        if "Successfully estimated coordinates:" in line:
            found_success_header = True
            continue
        if found_success_header:
            float_match = re.search(r"([-+]?[0-9]*\.?[0-9]+)$", line)
            if "Latitude" in line and float_match:
                est_lat = float(float_match.group(1))
            elif "Longitude" in line and float_match:
                est_lon = float(float_match.group(1))
            if est_lat is not None and est_lon is not None:
                break
    return est_lat, est_lon

def create_geojson_from_bbox_coords(min_lon, min_lat, max_lon, max_lat, properties=None):
    """Creates a GeoJSON FeatureCollection string from EPSG:4326 bounding box coordinates."""
    if properties is None:
        properties = {}
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat],
                        [min_lon, max_lat],
                        [max_lon, max_lat],
                        [max_lon, min_lat],
                        [min_lon, min_lat]
                    ]]
                }
            }
        ]
    }

def fetch_refined_search_image(
    center_lon_4326: float, center_lat_4326: float,
    geo_width_deg: float, geo_height_deg: float,
    image_pixels: int, output_path: Path
) -> tuple[Path | None, tuple[float, float, float, float] | None]:
    """
    Fetches a refined search satellite image centered around the given coords.
    Returns the path to the image and its EPSG:4326 bounds (min_lon, min_lat, max_lon, max_lat).
    """
    min_lon_4326 = center_lon_4326 - geo_width_deg / 2
    max_lon_4326 = center_lon_4326 + geo_width_deg / 2
    min_lat_4326 = center_lat_4326 - geo_height_deg / 2
    max_lat_4326 = center_lat_4326 + geo_height_deg / 2
    
    refined_bbox_4326_bounds = (min_lon_4326, min_lat_4326, max_lon_4326, max_lat_4326)

    try:
        bbox_3035 = convert_4326_bbox_to_3035_bbox(*refined_bbox_4326_bounds)
    except Exception as e:
        print(f"  Error converting refined bbox to EPSG:3035: {e}")
        return None, None

    params = {
        "bbox": bbox_3035.to_query_string(),
        "bboxSR": bbox_3035.projection.value.split(':')[1],
        "size": f"{image_pixels},{image_pixels}",
        "imageSR": bbox_3035.projection.value.split(':')[1],
        "format": "jpeg", "f": "image",
    }
    print(f"  Fetching refined search image. API Params: {params}")
    try:
        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.save(output_path)
        print(f"  Saved refined search image to {output_path}")
        return output_path, refined_bbox_4326_bounds
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching refined search image: {e}")
        if hasattr(e, 'response') and e.response is not None: print(f"  Response content: {e.response.text}")
        return None, None
    except Exception as e:
        print(f"  Error saving refined search image: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Run direct localization with refined search for a batch of images.")
    parser.add_argument("truth_csv", help="Path to the input truth CSV (image_id,latitude,longitude).")
    parser.add_argument("target_images_base_dir", help="Base directory where target images are located.")
    parser.add_argument("output_estimations_csv", help="Path to save the output estimations CSV.")
    parser.add_argument("--refined_search_image_dir", default="data/example/refined_search_images/", help="Directory to save fetched refined search satellite images.")
    parser.add_argument("--target_image_dim", type=int, default=ASSUMED_TARGET_IMG_DIM, help="Dimension (width/height) of the square target UAV images in pixels.")
    parser.add_argument("--refined_search_geo_width", type=float, default=DEFAULT_REFINED_SEARCH_GEO_WIDTH_DEGREES, help="Geographic width (longitude degrees) of the refined search area.")
    parser.add_argument("--refined_search_geo_height", type=float, default=DEFAULT_REFINED_SEARCH_GEO_HEIGHT_DEGREES, help="Geographic height (latitude degrees) of the refined search area.")
    parser.add_argument("--refined_search_pixels", type=int, default=DEFAULT_REFINED_SEARCH_IMAGE_PIXELS, help="Pixel dimension for the fetched refined search satellite image.")

    args = parser.parse_args()

    DEFAULT_SUPERGLUE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.refined_search_image_dir).mkdir(parents=True, exist_ok=True)
    
    all_estimations = []

    try:
        truth_df = pd.read_csv(args.truth_csv)
    except FileNotFoundError:
        print(f"Error: Truth CSV not found at {args.truth_csv}")
        return

    for index, row in truth_df.iterrows():
        image_id_filename = row['image_id']
        true_lat = float(row['latitude'])
        true_lon = float(row['longitude'])
        
        target_image_full_path = Path(args.target_images_base_dir) / image_id_filename

        if not target_image_full_path.exists():
            print(f"Warning: Target image {target_image_full_path} not found. Skipping.")
            all_estimations.append({"image_id": image_id_filename, "latitude": "", "longitude": ""})
            continue
            
        print(f"\nProcessing {image_id_filename} (True Lat: {true_lat:.6f}, Lon: {true_lon:.6f})...")

        # 1. Fetch refined satellite image for this target
        refined_satellite_image_output_path = Path(args.refined_search_image_dir) / f"{target_image_full_path.stem}_refined_satellite.jpg"
        
        fetched_refined_path, refined_bbox_4326 = fetch_refined_search_image(
            true_lon, true_lat,
            args.refined_search_geo_width, args.refined_search_geo_height,
            args.refined_search_pixels,
            refined_satellite_image_output_path
        )

        if not fetched_refined_path or not refined_bbox_4326:
            print(f"  Failed to fetch refined satellite image for {image_id_filename}. Skipping.")
            all_estimations.append({"image_id": image_id_filename, "latitude": "", "longitude": ""})
            continue
        
        current_refined_satellite_image_path_str = str(fetched_refined_path.resolve())
        
        # 2. Create temporary GeoJSON for the refined search area
        temp_geojson_data = create_geojson_from_bbox_coords(*refined_bbox_4326)
        temp_geojson_filename = f"{target_image_full_path.stem}_refined_search_area.json"
        temp_geojson_path = DEFAULT_SUPERGLUE_OUTPUT_DIR / temp_geojson_filename
        with open(temp_geojson_path, 'w') as f_json:
            json.dump(temp_geojson_data, f_json)
        
        # 3. Create pair file for SuperGlue
        pair_file_name = f"{target_image_full_path.stem}_vs_{Path(current_refined_satellite_image_path_str).stem}_pair.txt"
        pair_file_path = DEFAULT_SUPERGLUE_OUTPUT_DIR / pair_file_name
        
        with open(pair_file_path, 'w') as f:
            f.write(f"{str(target_image_full_path.resolve())} {current_refined_satellite_image_path_str}\n")

        # 4. Define stems for SuperGlue output NPZ file
        stem0 = target_image_full_path.stem
        stem1 = Path(current_refined_satellite_image_path_str).stem
        npz_matches_path = DEFAULT_SUPERGLUE_OUTPUT_DIR / f"{stem0}_{stem1}_matches.npz"

        # 5. Run SuperGlue
        sg_cmd = [
            "python", DEFAULT_SUPERGLUE_SCRIPT_PATH,
            "--input_pairs", str(pair_file_path),
            "--input_dir", str(project_root), # SuperGlue needs to resolve paths from project root
            "--output_dir", str(DEFAULT_SUPERGLUE_OUTPUT_DIR),
            "--superglue", "outdoor", "--resize", "-1" 
        ]
        success_sg, _ = run_command(sg_cmd)

        est_lat, est_lon = None, None
        if success_sg and npz_matches_path.exists():
            # 6. Run estimation
            est_cmd = [
                "python", DEFAULT_ESTIMATOR_SCRIPT_PATH,
                str(npz_matches_path),
                str(temp_geojson_path), # Use the temporary GeoJSON for this refined search
                str(args.refined_search_pixels), # Width of the refined satellite image
                str(args.refined_search_pixels), # Height of the refined satellite image
                str(args.target_image_dim),      # Width of the UAV/target image
                str(args.target_image_dim)       # Height of the UAV/target image
            ]
            success_est, out_est = run_command(est_cmd)
            if success_est:
                est_lat, est_lon = parse_estimation_output_for_coords(out_est)
        else:
            print(f"  SuperGlue match failed or NPZ not found for {image_id_filename}.")
        
        all_estimations.append({"image_id": image_id_filename, "latitude": est_lat, "longitude": est_lon})
        
        # Clean up temporary GeoJSON
        # os.remove(temp_geojson_path) # Optional: keep for debugging

    # Write all estimations to CSV
    if all_estimations:
        fieldnames = ["image_id", "latitude", "longitude"]
        with open(args.output_estimations_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row_data in all_estimations:
                row_data_formatted = {
                    "image_id": row_data["image_id"],
                    "latitude": f"{row_data['latitude']:.8f}" if row_data['latitude'] is not None else "",
                    "longitude": f"{row_data['longitude']:.8f}" if row_data['longitude'] is not None else ""
                }
                writer.writerow(row_data_formatted)
        print(f"\nBatch direct localization with refined search complete. Estimations saved to {args.output_estimations_csv}")
    else:
        print("No estimations to save.")

if __name__ == '__main__':
    main()
