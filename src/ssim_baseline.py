import json
import cv2
import numpy as np
from src.utility.get_satellite_image import get_satellite_image
from skimage.metrics import structural_similarity as ssim

def ssim_baseline(uav_image_path, search_area_json_path):
    """
    A baseline for relocalisation that uses Structural Similarity Index (SSIM).
    """

    # Load the search area from the JSON file
    with open(search_area_json_path, 'r') as f:
        search_area = json.load(f)

    # Extract the coordinates of the bounding box
    coordinates = search_area['features'][0]['geometry']['coordinates'][0]
    min_lon = min(c[0] for c in coordinates)
    min_lat = min(c[1] for c in coordinates)
    max_lon = max(c[0] for c in coordinates)
    max_lat = max(c[1] for c in coordinates)

    # Load the UAV image
    uav_image = cv2.imread(uav_image_path, cv2.IMREAD_GRAYSCALE)
    if uav_image is None:
        raise ValueError(f"Could not read UAV image at {uav_image_path}")

    # Get a satellite image
    satellite_image = get_satellite_image(min_lon, min_lat, max_lon, max_lat)
    if satellite_image is None:
        raise ValueError("Could not retrieve satellite image")

    satellite_image = cv2.cvtColor(satellite_image, cv2.COLOR_BGR2GRAY)

    # Resize satellite image to match UAV image dimensions
    satellite_image = cv2.resize(satellite_image, (uav_image.shape[1], uav_image.shape[0]))

    # Apply histogram equalization
    uav_image = cv2.equalizeHist(uav_image)
    satellite_image = cv2.equalizeHist(satellite_image)

    # Calculate the SSIM
    similarity_score, _ = ssim(uav_image, satellite_image, full=True)

    print(f"SSIM score: {similarity_score}")

    # Return the center coordinates of the satellite image as the predicted location
    predicted_lon = (min_lon + max_lon) / 2
    predicted_lat = (min_lat + max_lat) / 2
    return predicted_lon, predicted_lat


if __name__ == '__main__':
    uav_image_path = 'data/example/images/rickmansworth_example.jpg'
    search_area_json_path = 'data/example/images/rickmansworth_example_search_area.json'
    predicted_lon, predicted_lat = ssim_baseline(uav_image_path, search_area_json_path)
    print(f"Predicted Longitude: {predicted_lon}, Predicted Latitude: {predicted_lat}")
