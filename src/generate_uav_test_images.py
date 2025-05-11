import json
import os
import random
import math
import csv
import requests # For API calls
from io import BytesIO # For handling image data from response
from pathlib import Path
from PIL import Image

# Add project root to sys.path to allow for 'from src.utility...' imports
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utility.bounding_box import BoundingBox, Projection
from src.utility.epsg_4326_to_3035 import convert_4326_bbox_to_3035_bbox

# Constants
API_URL = "https://image.discomap.eea.europa.eu/arcgis/rest/services/GioLand/VHR_2018_LAEA/ImageServer/exportImage"
BASE_GEOJSON_PATH = Path("data/example/images/eiffel_tower_search_area.json") # Used to define overall area for random centers
OUTPUT_DIR = Path("data/example/images/test/")
TRUTH_CSV_PATH = OUTPUT_DIR / "test_images_truth.csv"
NUM_IMAGES = 3
UAV_IMAGE_REQUEST_SIZE_PIXELS = 64  # Request 512x512 images from API for the small bbox
# Define the approximate geographic size of the high-resolution "UAV" image
# These values are in degrees.
# For Paris (approx lat 48.85°):
# 1 degree latitude ~ 111.32 km
# 1 degree longitude ~ 111.32 km * cos(48.85°) ~ 73.2 km
# So, 0.0005° lon ~ 36.6 meters
# And 0.00035° lat ~ 38.9 meters
# This gives a roughly 35-40m square area, which should be "street level" when rendered at 512x512px
UAV_TARGET_GEO_WIDTH_DEGREES = 0.0005  # Approx width of the desired high-res image area
UAV_TARGET_GEO_HEIGHT_DEGREES = 0.00035 # Approx height of the desired high-res image area

# Max random offset from the center of BASE_GEOJSON_PATH to pick UAV image centers (in degrees)
# The Eiffel Tower search area is roughly 0.03 degrees wide and 0.02 degrees tall.
# An offset of 0.005 degrees is about 1/6th to 1/4th of that, allowing varied locations.
MAX_RANDOM_OFFSET_DEGREES = 0.005


def get_geojson_bounds_4326(geojson_path: Path) -> tuple[float, float, float, float]:
    """Reads a GeoJSON file and returns its overall bounding box in EPSG:4326."""
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    all_coords = []
    for feature in data['features']:
        geom = feature['geometry']
        if geom['type'] == 'Polygon':
            # Takes the first ring of coordinates
            all_coords.extend(geom['coordinates'][0])
        elif geom['type'] == 'MultiPolygon':
            for poly in geom['coordinates']:
                all_coords.extend(poly[0])
    
    if not all_coords:
        raise ValueError("No coordinates found in GeoJSON.")

    min_lon = min(c[0] for c in all_coords)
    max_lon = max(c[0] for c in all_coords)
    min_lat = min(c[1] for c in all_coords)
    max_lat = max(c[1] for c in all_coords)
    
    return min_lon, min_lat, max_lon, max_lat

def fetch_high_res_image(bbox_3035: BoundingBox, width_px: int, height_px: int) -> Image.Image | None:
    """Fetches an image from the API for the given EPSG:3035 bounding box."""
    params = {
        "bbox": bbox_3035.to_query_string(),
        "bboxSR": bbox_3035.projection.value.split(':')[1],
        "size": f"{width_px},{height_px}",
        "imageSR": bbox_3035.projection.value.split(':')[1],
        "format": "jpeg",
        "f": "image",
    }
    print(f"  Requesting image from API. URL: {API_URL}, Params: {params}")
    try:
        response = requests.get(API_URL, params=params, timeout=30) # Added timeout
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching image: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response content: {e.response.text}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred during image fetch: {e}")
        return None


def generate_images():
    """Generates high-resolution UAV test images by fetching them from an API."""
    print(f"Attempting to generate {NUM_IMAGES} high-resolution UAV images...")
    if not BASE_GEOJSON_PATH.exists():
        print(f"Error: Base GeoJSON for area definition not found at {BASE_GEOJSON_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Truth CSV: {TRUTH_CSV_PATH.resolve()}")

    try:
        main_min_lon, main_min_lat, main_max_lon, main_max_lat = get_geojson_bounds_4326(BASE_GEOJSON_PATH)
    except Exception as e:
        print(f"Error reading base GeoJSON {BASE_GEOJSON_PATH}: {e}")
        return
    
    main_center_lon = (main_min_lon + main_max_lon) / 2
    main_center_lat = (main_min_lat + main_max_lat) / 2
    print(f"Base area EPSG:4326 bounds: Lon({main_min_lon:.5f}-{main_max_lon:.5f}), Lat({main_min_lat:.5f}-{main_max_lat:.5f})")
    print(f"Base area EPSG:4326 center: Lon({main_center_lon:.5f}), Lat({main_center_lat:.5f})")

    generated_truths = []

    for i in range(NUM_IMAGES):
        print(f"\nGenerating image {i+1}/{NUM_IMAGES}:")

        # 1. Determine random target center in EPSG:4326 within the base area + offset
        offset_lon = random.uniform(-MAX_RANDOM_OFFSET_DEGREES, MAX_RANDOM_OFFSET_DEGREES)
        offset_lat = random.uniform(-MAX_RANDOM_OFFSET_DEGREES, MAX_RANDOM_OFFSET_DEGREES)
        
        # Ensure the center is within the broader bounds of the original GeoJSON
        # This is a simplified way to keep it roughly within the initial large search area
        target_center_lon_4326 = main_center_lon + offset_lon
        target_center_lat_4326 = main_center_lat + offset_lat
        
        target_center_lon_4326 = max(main_min_lon, min(target_center_lon_4326, main_max_lon))
        target_center_lat_4326 = max(main_min_lat, min(target_center_lat_4326, main_max_lat))

        print(f"  Target UAV center (EPSG:4326): Lon={target_center_lon_4326:.6f}, Lat={target_center_lat_4326:.6f}")

        # 2. Define small EPSG:4326 bounding box for this UAV image
        uav_min_lon_4326 = target_center_lon_4326 - UAV_TARGET_GEO_WIDTH_DEGREES / 2
        uav_max_lon_4326 = target_center_lon_4326 + UAV_TARGET_GEO_WIDTH_DEGREES / 2
        uav_min_lat_4326 = target_center_lat_4326 - UAV_TARGET_GEO_HEIGHT_DEGREES / 2
        uav_max_lat_4326 = target_center_lat_4326 + UAV_TARGET_GEO_HEIGHT_DEGREES / 2
        print(f"  UAV BBox (EPSG:4326): Lon({uav_min_lon_4326:.6f}-{uav_max_lon_4326:.6f}), Lat({uav_min_lat_4326:.6f}-{uav_max_lat_4326:.6f})")

        # 3. Convert UAV BBox to EPSG:3035
        try:
            uav_bbox_3035 = convert_4326_bbox_to_3035_bbox(
                uav_min_lon_4326, uav_min_lat_4326, uav_max_lon_4326, uav_max_lat_4326
            )
            print(f"  UAV BBox (EPSG:3035): {uav_bbox_3035.to_query_string()}")
        except Exception as e:
            print(f"  Error converting bbox to EPSG:3035: {e}")
            continue
            
        # 4. Fetch the high-resolution image
        image = fetch_high_res_image(uav_bbox_3035, UAV_IMAGE_REQUEST_SIZE_PIXELS, UAV_IMAGE_REQUEST_SIZE_PIXELS)

        if image:
            # 5. Save image
            output_filename = f"uav_eiffel_highres_test_{i+1:02d}.jpg"
            output_path = OUTPUT_DIR / output_filename
            try:
                image.save(output_path, quality=95)
                print(f"  Saved UAV image to: {output_path.resolve()}")
                generated_truths.append({
                    "image_id": output_filename,
                    "latitude": f"{target_center_lat_4326:.8f}", # Truth is the target center
                    "longitude": f"{target_center_lon_4326:.8f}"
                })
            except Exception as e:
                print(f"  Error saving UAV image: {e}")
        else:
            print(f"  Failed to fetch image {i+1}. Skipping.")

    # Append to or create truth CSV
    if generated_truths:
        file_exists = TRUTH_CSV_PATH.exists()
        try:
            with open(TRUTH_CSV_PATH, 'a', newline='') as csvfile:
                fieldnames = ["image_id", "latitude", "longitude"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists or TRUTH_CSV_PATH.stat().st_size == 0:
                    writer.writeheader() # Write header only if file is new or empty
                for truth_data in generated_truths:
                    writer.writerow(truth_data)
            print(f"\nSuccessfully appended {len(generated_truths)} new entries to truth CSV: {TRUTH_CSV_PATH.resolve()}")
        except Exception as e:
            print(f"Error writing to truth CSV: {e}")
    else:
        print("\nNo new images were successfully generated to add to the truth CSV.")

if __name__ == "__main__":
    generate_images()
