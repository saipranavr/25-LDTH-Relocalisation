import sys
sys.path.append('/Users/pranavreddy/Documents/GitHub/25-LDTH-Relocalisation')
import json
import cv2
import numpy as np
from src.utility.get_satellite_image import get_satellite_image

def naive_baseline(uav_image_path, search_area_json_path):
    """
    A naive baseline for relocalisation that compares the UAV image with a satellite image
    within the provided search area using average pixel value difference.
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
    uav_image = cv2.imread(uav_image_path)
    if uav_image is None:
        raise ValueError(f"Could not read UAV image at {uav_image_path}")

    # Get a satellite image
    print(f"Getting satellite image for: min_lon={min_lon}, min_lat={min_lat}, max_lon={max_lon}, max_lat={max_lat}")
    satellite_image = get_satellite_image(min_lon, min_lat, max_lon, max_lat)
    if satellite_image is None:
        raise ValueError("Could not retrieve satellite image")
    print("Satellite image retrieved successfully")

    # Resize satellite image to match UAV image dimensions
    satellite_image = cv2.resize(satellite_image, (uav_image.shape[1], uav_image.shape[0]))

    # Calculate the average pixel value difference
    uav_avg = np.mean(uav_image)
    satellite_avg = np.mean(satellite_image)
    similarity_score = abs(uav_avg - satellite_avg)

    print(f"Similarity score: {similarity_score}")

    # Return the center coordinates of the satellite image as the predicted location
    predicted_lon = (min_lon + max_lon) / 2
    predicted_lat = (min_lat + max_lat) / 2
    return predicted_lon, predicted_lat, similarity_score


if __name__ == '__main__':
    uav_image_path = 'data/example/images/rickmansworth_example.jpg'
    search_area_json_path = 'data/example/images/rickmansworth_example_search_area.json'
    predicted_lon, predicted_lat, similarity_score = naive_baseline(uav_image_path, search_area_json_path)
    print(f"Predicted Longitude: {predicted_lon}, Predicted Latitude: {predicted_lat}")
