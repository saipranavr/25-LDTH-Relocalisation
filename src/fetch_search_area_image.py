import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import argparse
import os
import sys # Add sys module
from pathlib import Path # Add Path module

# Add project root to sys.path to allow for 'from src.utility...' imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utility.epsg_4326_to_3035 import geojson_to_3035_bboxes
from src.utility.bounding_box import BoundingBox # Assuming BoundingBox is needed by geojson_to_3035_bboxes or for type hinting

# API URL for VHR 2018 imagery (known to be working)
API_URL = "https://image.discomap.eea.europa.eu/arcgis/rest/services/GioLand/VHR_2018_LAEA/ImageServer/exportImage"

def fetch_image_for_geojson_area(geojson_path: str, output_image_path: str, image_pixels_w: int = 1024, image_pixels_h: int = 1024, display_image: bool = True):
    """
    Fetches a satellite image for the bounding box defined in a GeoJSON file.

    Args:
        geojson_path: Path to the GeoJSON file defining the search area.
        output_image_path: Path to save the fetched satellite image.
        image_pixels_w: Width of the requested image in pixels.
        image_pixels_h: Height of the requested image in pixels.
        display_image: Whether to display the image after fetching.
    """
    print(f"Processing GeoJSON: {geojson_path}")

    # 1. Get EPSG:3035 bounding box(es) from GeoJSON
    # Assuming the GeoJSON contains one feature defining the overall search area
    bboxes_3035 = geojson_to_3035_bboxes(geojson_path)
    if not bboxes_3035:
        print(f"Error: No bounding boxes could be derived from {geojson_path}")
        return
    
    # For this script, we assume the first bounding box is the one we want for the entire area.
    # If the GeoJSON could define multiple disjoint areas, more complex logic would be needed.
    search_area_bbox_3035: BoundingBox = bboxes_3035[0]
    
    print(f"Target Bounding Box (EPSG:3035): {search_area_bbox_3035.to_query_string()}")

    # 2. Prepare API request parameters
    params = {
        "bbox": search_area_bbox_3035.to_query_string(),
        "bboxSR": search_area_bbox_3035.projection.value.split(':')[1], # API expects "3035"
        "size": f"{image_pixels_w},{image_pixels_h}",
        "imageSR": search_area_bbox_3035.projection.value.split(':')[1],
        "format": "jpeg", # Using jpeg for potentially smaller file size for large images
        "f": "image",
    }

    # 3. Fetch the image
    print(f"Requesting large area image from API with params: {params}")
    response = requests.get(API_URL, params=params)
    print("Full Request URL:", response.url)

    try:
        response.raise_for_status()  # Raise an exception for HTTP errors
        image = Image.open(BytesIO(response.content))
        
        # 4. Save the image
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            image.save(output_image_path)
            print(f"Large area image successfully saved to {output_image_path}")
        except Exception as e:
            print(f"Error saving image to {output_image_path}: {e}")
            return # Don't proceed if save fails

        # 5. Display the image (optional)
        if display_image:
            plt.imshow(image) # type: ignore
            plt.title(f"Search Area Image: {os.path.basename(geojson_path)}\nSize: {image_pixels_w}x{image_pixels_h}px")
            plt.axis('off') # type: ignore
            plt.show() # type: ignore

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        print("Response content:", response.content.decode(errors='ignore'))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch a large satellite image for a GeoJSON defined search area.")
    parser.add_argument("geojson_file", type=str, help="Path to the input GeoJSON file defining the search area.")
    parser.add_argument("output_image_file", type=str, help="Path to save the fetched satellite image (e.g., data/example/images/search_area.jpg).")
    parser.add_argument("--width", type=int, default=1024, help="Width of the output image in pixels.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the output image in pixels.")
    parser.add_argument("--no-display", action="store_false", dest="display", help="Do not display the image after fetching.")
    
    args = parser.parse_args()

    fetch_image_for_geojson_area(
        geojson_path=args.geojson_file,
        output_image_path=args.output_image_file,
        image_pixels_w=args.width,
        image_pixels_h=args.height,
        display_image=args.display
    )
