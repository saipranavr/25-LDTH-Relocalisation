# import requests # No longer needed for direct API calls
from PIL import Image
# from io import BytesIO # No longer needed if staticmap returns PIL Image
# import matplotlib.pyplot as plt # Not used in this script for display
# from pyproj import Transformer # No longer needed for EPSG:3035 conversion
import argparse
import os
import random
import json
import sys
from pathlib import Path
import math # For cos and radians

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# from src.utility.bounding_box import BoundingBox, Projection # No longer needed
from staticmap import StaticMap # For fetching static map images
# from staticmap.util import guess_zoom_for_bounds # To calculate zoom level - apparently not found

# URL for a free satellite imagery tile server (Esri World Imagery)
ESRI_WORLD_IMAGERY_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def fetch_and_save_image(
    lat_4326: float,
    lon_4326: float,
    image_id_base: str,
    index: int,
    output_dir: str,
    box_size_meters: float = 500,
    image_pixels: int = 512
    ):
    """
    Fetches a satellite image for given EPSG:4326 coordinates using staticmap
    and saves it.
    """
    output_filename = f"{image_id_base}_{index:02d}.jpg"
    output_image_path = os.path.join(output_dir, output_filename)

    print(f"Fetching image for Lat: {lat_4326:.6f}, Lon: {lon_4326:.6f} (approx {box_size_meters}m box) -> {output_filename}")

    # 1. Calculate WGS84 bounding box from center point and box_size_meters
    # Approximate conversion: 1 degree latitude ~= 111.111 km
    # 1 degree longitude ~= 111.111 km * cos(latitude)
    half_side_meters = box_size_meters / 2.0
    
    delta_lat_deg = half_side_meters / 111111.0
    delta_lon_deg = half_side_meters / (111111.0 * math.cos(math.radians(lat_4326)))

    min_lat = lat_4326 - delta_lat_deg
    max_lat = lat_4326 + delta_lat_deg
    min_lon = lon_4326 - delta_lon_deg
    max_lon = lon_4326 + delta_lon_deg

    # 2. Create StaticMap object
    static_map_obj = StaticMap(
        width=image_pixels,
        height=image_pixels,
        url_template=ESRI_WORLD_IMAGERY_URL
    )

    # 3. Render the image for the bounding box
    try:
        # Calculate the center of the previously determined bounding box
        # This bounding box was derived from lat_4326, lon_4326 and box_size_meters
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0
        
        # Use a fixed zoom level. Zoom level 17 is typically good for ~500m areas.
        # The box_size_meters argument is now primarily for centering.
        # Actual ground coverage will depend on this zoom level and pixel dimensions.
        fixed_zoom_level = 17 
        
        print(f"  Requesting image from tile server for center: ({center_lon:.6f}, {center_lat:.6f}), Zoom: {fixed_zoom_level}")
        image = static_map_obj.render(center=(center_lon, center_lat), zoom=fixed_zoom_level)
        
        # 4. Save the image
        os.makedirs(output_dir, exist_ok=True)
        image.save(output_image_path)
        print(f"  Successfully saved to {output_image_path}")
        return True
    except ImportError:
        print("Error: The 'staticmap' library is not installed. Please install it (e.g., pip install staticmap).")
        return False
    except Exception as e:
        print(f"  An error occurred while fetching/saving {output_filename}: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_bounds_from_geojson(geojson_path: str):
    """Reads a GeoJSON file and extracts the overall lat/lon bounding box."""
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading GeoJSON file {geojson_path}: {e}")
        return None

    # Assuming the first feature and first polygon coordinates
    try:
        coords = data['features'][0]['geometry']['coordinates'][0]
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
        return min(lats), max(lats), min(lons), max(lons)
    except (IndexError, KeyError, TypeError) as e:
        print(f"Error parsing GeoJSON structure in {geojson_path}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random test images within a GeoJSON defined search area.")
    parser.add_argument("search_area_geojson", help="Path to the GeoJSON file defining the search area boundaries.")
    parser.add_argument("output_directory", help="Directory to save the generated test images.")
    parser.add_argument("--num_images", type=int, default=20, help="Number of random images to generate.")
    parser.add_argument("--img_size_meters", type=float, default=500, help="Side length of the square bounding box for each image in meters.")
    parser.add_argument("--img_pixels", type=int, default=512, help="Pixel dimension (width and height) for each fetched image.")
    parser.add_argument("--base_filename", type=str, default="test_image", help="Base name for the output image files.")
    
    args = parser.parse_args()

    bounds = get_bounds_from_geojson(args.search_area_geojson)
    if not bounds:
        sys.exit(1)
    
    min_lat, max_lat, min_lon, max_lon = bounds
    print(f"Search Area Bounds (EPSG:4326): Lat [{min_lat:.6f}, {max_lat:.6f}], Lon [{min_lon:.6f}, {max_lon:.6f}]")

    generated_count = 0
    truth_data = [] # To store (filename, lat, lon)

    for i in range(args.num_images):
        rand_lat = random.uniform(min_lat, max_lat)
        rand_lon = random.uniform(min_lon, max_lon)
        
        # Construct filename to store in CSV, relative to output_directory
        image_filename = f"{args.base_filename}_{i+1:02d}.jpg"

        success = fetch_and_save_image(
            lat_4326=rand_lat,
            lon_4326=rand_lon,
            image_id_base=args.base_filename, # fetch_and_save_image will form the full path
            index=i + 1,
            output_dir=args.output_directory,
            box_size_meters=args.img_size_meters,
            image_pixels=args.img_pixels
        )
        if success:
            generated_count += 1
            truth_data.append({"image_id": image_filename, "latitude": rand_lat, "longitude": rand_lon})
        
        # Optional: add a small delay if hitting API too fast
        # import time
        # time.sleep(0.5) 

    print(f"\nFinished generating images. Successfully created {generated_count}/{args.num_images} images in {args.output_directory}")

    # Write truth data to CSV
    if truth_data:
        csv_output_path = os.path.join(args.output_directory, "test_images_truth.csv")
        try:
            with open(csv_output_path, 'w') as f:
                f.write("image_id,latitude,longitude\n") # Header
                for item in truth_data:
                    f.write(f"{item['image_id']},{item['latitude']:.8f},{item['longitude']:.8f}\n")
            print(f"Successfully wrote truth data to {csv_output_path}")
        except Exception as e:
            print(f"Error writing truth CSV to {csv_output_path}: {e}")
    else:
        print("No images were successfully generated, so no truth CSV created.")
