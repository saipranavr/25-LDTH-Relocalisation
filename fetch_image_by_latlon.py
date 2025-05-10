import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from pyproj import Transformer

from src.utility.bounding_box import BoundingBox, Projection

# Copernicus API URL for VHR 2021 imagery
API_URL = "https://image.discomap.eea.europa.eu/arcgis/rest/services/GioLand/VHR_2018_LAEA/ImageServer/exportImage"

def fetch_and_display_image_at_coords(lat_4326: float, lon_4326: float, image_id: str, box_size_meters: float = 500, image_pixels: int = 512):
    """
    Fetches a satellite image for given EPSG:4326 coordinates, saves it, and displays it.

    Args:
        lat_4326: Latitude in EPSG:4326.
        lon_4326: Longitude in EPSG:4326.
        image_id: The base name for the saved image file (e.g., "eiffel_tower_example").
        box_size_meters: The side length of the square bounding box in meters (EPSG:3035).
        image_pixels: The dimension (width and height) of the requested image in pixels.
    """
    print(f"Fetching image for Lat: {lat_4326}, Lon: {lon_4326}, ID: {image_id}")
    output_image_path = f"data/example/images/{image_id}.jpg"

    # 1. Transform coordinates from EPSG:4326 to EPSG:3035
    transformer = Transformer.from_crs(Projection.EPSG_4326.value, Projection.EPSG_3035.value, always_xy=True)
    center_lon_3035, center_lat_3035 = transformer.transform(lon_4326, lat_4326)
    print(f"Transformed center (EPSG:3035): Lon={center_lon_3035}, Lat={center_lat_3035}")

    # 2. Define the bounding box in EPSG:3035
    half_side = box_size_meters / 2
    bbox_3035 = BoundingBox(
        min_lat=int(center_lat_3035 - half_side),
        max_lat=int(center_lat_3035 + half_side),
        min_lon=int(center_lon_3035 - half_side),
        max_lon=int(center_lon_3035 + half_side),
        projection=Projection.EPSG_3035
    )
    print(f"Bounding Box (EPSG:3035): {bbox_3035.to_query_string()}")

    # 3. Prepare API request parameters
    params = {
        "bbox": bbox_3035.to_query_string(),
        "bboxSR": bbox_3035.projection.value.split(':')[1], # API expects "3035", not "EPSG:3035"
        "size": f"{image_pixels},{image_pixels}",
        "imageSR": bbox_3035.projection.value.split(':')[1],
        "format": "png",
        "f": "image",
    }

    # 4. Fetch the image
    print(f"Requesting image from API with params: {params}")
    response = requests.get(API_URL, params=params)
    print("Full Request URL:", response.url)

    try:
        response.raise_for_status()  # Raise an exception for HTTP errors
        image = Image.open(BytesIO(response.content))
        
        # 5. Save the image
        try:
            image.save(output_image_path)
            print(f"Image successfully saved to {output_image_path}")
        except Exception as e:
            print(f"Error saving image to {output_image_path}: {e}")

        # 6. Display the image
        plt.imshow(image) # type: ignore
        plt.title(f"Satellite Image at ({lat_4326:.6f}, {lon_4326:.6f})\nID: {image_id}, Box: {box_size_meters}m, Size: {image_pixels}px")
        plt.xlabel(f"EPSG:3035 Longitude (Center: {center_lon_3035:.2f})")
        plt.ylabel(f"EPSG:3035 Latitude (Center: {center_lat_3035:.2f})")
        # Display bounding box coordinates on axes for context
        tick_lons = [bbox_3035.min_lon, center_lon_3035, bbox_3035.max_lon]
        tick_lats = [bbox_3035.min_lat, center_lat_3035, bbox_3035.max_lat]
        plt.xticks(ticks=[0, image_pixels//2, image_pixels-1], labels=[f"{val:.0f}" for val in tick_lons]) # type: ignore
        plt.yticks(ticks=[0, image_pixels//2, image_pixels-1], labels=[f"{val:.0f}" for val in tick_lats]) # type: ignore
        plt.show() # type: ignore

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        print("Response content:", response.content.decode(errors='ignore'))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # New example: Eiffel Tower, freshly fetched for comparison
    new_image_id = "eiffel_tower_freshly_fetched" # Changed for this test
    target_latitude = 48.8584  # Eiffel Tower latitude
    target_longitude = 2.2945  # Eiffel Tower longitude
    
    fetch_and_display_image_at_coords(target_latitude, target_longitude, new_image_id, box_size_meters=500)
