import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt # For optional display, though likely off for batch
from pyproj import Transformer
import argparse
import os
import random
import json
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utility.bounding_box import BoundingBox, Projection

# API URL for VHR 2018 imagery (known to be working)
API_URL = "https://image.discomap.eea.europa.eu/arcgis/rest/services/GioLand/VHR_2018_LAEA/ImageServer/exportImage"

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
    Fetches a satellite image for given EPSG:4326 coordinates and saves it.
    Adapted from fetch_image_by_latlon.py
    """
    output_filename = f"{image_id_base}_{index:02d}.jpg"
    output_image_path = os.path.join(output_dir, output_filename)

    print(f"Fetching image for Lat: {lat_4326:.6f}, Lon: {lon_4326:.6f} -> {output_filename}")

    # 1. Transform coordinates from EPSG:4326 to EPSG:3035
    transformer = Transformer.from_crs(Projection.EPSG_4326.value, Projection.EPSG_3035.value, always_xy=True)
    center_lon_3035, center_lat_3035 = transformer.transform(lon_4326, lat_4326)

    # 2. Define the bounding box in EPSG:3035
    half_side = box_size_meters / 2
    bbox_3035 = BoundingBox(
        min_lat=int(center_lat_3035 - half_side),
        max_lat=int(center_lat_3035 + half_side),
        min_lon=int(center_lon_3035 - half_side),
        max_lon=int(center_lon_3035 + half_side),
        projection=Projection.EPSG_3035
    )

    # 3. Prepare API request parameters
    params = {
        "bbox": bbox_3035.to_query_string(),
        "bboxSR": bbox_3035.projection.value.split(':')[1],
        "size": f"{image_pixels},{image_pixels}",
        "imageSR": bbox_3035.projection.value.split(':')[1],
        "format": "jpeg", 
        "f": "image",
    }

    # 4. Fetch the image
    response = requests.get(API_URL, params=params)

    try:
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        
        # 5. Save the image
        os.makedirs(output_dir, exist_ok=True)
        image.save(output_image_path)
        print(f"  Successfully saved to {output_image_path}")
        return True
    except requests.exceptions.HTTPError as err:
        print(f"  HTTP error for {output_filename}: {err}")
        print(f"  Response content: {response.content.decode(errors='ignore')}")
        return False
    except Exception as e:
        print(f"  An error occurred for {output_filename}: {e}")
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
