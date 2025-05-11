import argparse
import os
import sys
from pathlib import Path
import json # For loading GeoJSON
from io import BytesIO # For handling image data if needed, though staticmap might return PIL Image directly

from PIL import Image
import matplotlib.pyplot as plt
from staticmap import StaticMap # For fetching static map images
# from staticmap.util import guess_zoom_for_bounds # To calculate zoom level - apparently not found
from shapely.geometry import shape # For parsing GeoJSON geometry

# Add project root to sys.path to allow for 'from src.utility...' imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# We no longer need these as we're not using the old API or EPSG:3035 directly for fetching
# from src.utility.epsg_4326_to_3035 import geojson_to_3035_bboxes
# from src.utility.bounding_box import BoundingBox

# URL for a free satellite imagery tile server (Esri World Imagery)
# Other options:
# Google Satellite: 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}' (Check ToS for automated use)
ESRI_WORLD_IMAGERY_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def fetch_image_for_geojson_area(geojson_path: str, output_image_path: str, image_pixels_w: int = 1024, image_pixels_h: int = 1024, display_image: bool = True):
    """
    Fetches a satellite image for the bounding box defined in a GeoJSON file
    using a tile server via the staticmap library.

    Args:
        geojson_path: Path to the GeoJSON file defining the search area.
        output_image_path: Path to save the fetched satellite image.
        image_pixels_w: Width of the requested image in pixels.
        image_pixels_h: Height of the requested image in pixels.
        display_image: Whether to display the image after fetching.
    """
    print(f"Processing GeoJSON: {geojson_path} to fetch image of size {image_pixels_w}x{image_pixels_h}")

    try:
        # 1. Load GeoJSON and get EPSG:4326 (WGS84) bounding box
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)

        if not geojson_data.get('features') or not geojson_data['features'][0].get('geometry'):
            print(f"Error: GeoJSON file {geojson_path} is invalid or has no features/geometry.")
            return

        # Assuming the first feature's geometry defines the area of interest
        geom = shape(geojson_data['features'][0]['geometry'])
        min_lon_4326, min_lat_4326, max_lon_4326, max_lat_4326 = geom.bounds
        
        print(f"Target Bounding Box (EPSG:4326): min_lon={min_lon_4326}, min_lat={min_lat_4326}, max_lon={max_lon_4326}, max_lat={max_lat_4326}")

        # 2. Create StaticMap object
        # Using a known free tile server. Ensure compliance with its ToS.
        static_map_obj = StaticMap(
            width=image_pixels_w,
            height=image_pixels_h,
            url_template=ESRI_WORLD_IMAGERY_URL
        )

        # 3. Render the image for the bounding box
        center_lon = (min_lon_4326 + max_lon_4326) / 2.0
        center_lat = (min_lat_4326 + max_lat_4326) / 2.0
        
        # TODO: The zoom level here is fixed. Ideally, it should be calculated
        # to fit the GeoJSON bounds into the image_pixels_w/h.
        # However, the utility staticmap.util.guess_zoom_for_bounds was not found.
        # Consider asking the user for a zoom level or updating the staticmap library
        # to a version that includes this utility (e.g., staticmap==0.5.0).
        # For now, using a default zoom that might be okay for city-level views.
        # This will likely NOT perfectly fit arbitrary GeoJSON extents.
        fixed_zoom_level_for_search_area = 12 # Placeholder, adjust as needed or make it an argument
        
        print(f"  Requesting image from tile server for center: ({center_lon:.6f}, {center_lat:.6f}), Zoom: {fixed_zoom_level_for_search_area} (NOTE: Zoom is fixed, may not fit GeoJSON extent perfectly)")
        image = static_map_obj.render(center=(center_lon, center_lat), zoom=fixed_zoom_level_for_search_area)
        
        # 4. Save the image
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_image_path)
            if output_dir: # Check if output_dir is not empty (i.e. not saving in current dir)
                 os.makedirs(output_dir, exist_ok=True)
            image.save(output_image_path)
            print(f"Satellite image successfully saved to {output_image_path}")
        except Exception as e:
            print(f"Error saving image to {output_image_path}: {e}")
            return

        # 5. Display the image (optional)
        if display_image:
            # Convert PIL image to array for matplotlib
            # No, plt.imshow can handle PIL images directly
            plt.imshow(image)
            plt.title(f"Search Area Image: {os.path.basename(geojson_path)}\nSize: {image_pixels_w}x{image_pixels_h}px (Source: Esri World Imagery)")
            plt.axis('off')
            plt.show()

    except FileNotFoundError:
        print(f"Error: GeoJSON file not found at {geojson_path}")
    except ImportError:
        print("Error: The 'staticmap' or 'shapely' library is not installed. Please install it (e.g., pip install staticmap shapely).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


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
