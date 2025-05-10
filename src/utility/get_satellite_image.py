import requests
from PIL import Image
from io import BytesIO
import numpy as np
from src.utility.bounding_box import BoundingBox, Projection
from src.utility.epsg_4326_to_3035 import epsg_4326_to_3035
import time

# High res 2021 satellite imagery from Copernicus
url = "https://image.discomap.eea.europa.eu/arcgis/rest/services/GioLand/VHR_2018_LAEA/ImageServer/exportImage"

def get_satellite_image(min_lon, min_lat, max_lon, max_lat, retries=3):
    """
    Retrieves a satellite image from Copernicus based on WSG84 coordinates.
    Includes a retry mechanism to handle server errors.
    """
    # Convert WSG84 coordinates (EPSG:4326) to EPSG:3035
    min_x, min_y = epsg_4326_to_3035(min_lon, min_lat)
    if min_x is None or min_y is None:
        print("Error: Could not convert min_lon, min_lat to EPSG:3035")
        return None

    max_x, max_y = epsg_4326_to_3035(max_lon, max_lat)
    if max_x is None or max_y is None:
        print("Error: Could not convert max_lon, max_lat to EPSG:3035")
        return None

    bbox = BoundingBox(
        min_lat=min_y,
        max_lat=max_y,
        min_lon=min_x,
        max_lon=max_x,
        projection=Projection.EPSG_3035
    )

    # Bounding box is in EPSG:3035 (since it's LAEA projection)
    # Request a larger image size for potentially higher resolution
    # If the server has higher resolution data, it should provide it.
    # If not, it might upscale, or return an error, or return what it can.
    requested_size_px = 1024 # Try to get a 1024x1024 image
    params = {
        "bbox": bbox.to_query_string(),
        "bboxSR": "3035",
        "size": f"{requested_size_px},{requested_size_px}", 
        "imageSR": "3035",
        "format": "png",  # can also be "tiff"
        "f": "image",
    }
    print(f"Requesting satellite image with size: {params['size']}")

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            # Convert the image to a NumPy array
            img_array = np.array(image)
            return img_array

        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
            print("Response content:", response.content)
            if attempt < retries - 1:
                print(f"Retrying in 5 seconds (attempt {attempt + 1}/{retries})...")
                time.sleep(5)
            else:
                print(f"Max retries reached. Could not retrieve satellite image.")
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    return None
