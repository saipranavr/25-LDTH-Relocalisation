import sys
sys.path.append('/Users/pranavreddy/Documents/GitHub/25-LDTH-Relocalisation')
import json
import cv2
import numpy as np
from src.utility.get_satellite_image import get_satellite_image

def visualize_feature_matching(uav_image_path, search_area_json_path):
    """
    Visualizes the feature matching between the UAV image and the satellite image.
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

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=2000)

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(uav_image, None)
    kp2, des2 = orb.detectAndCompute(satellite_image, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)[:50]  # Limit to top 50 matches

    # Draw first 50 matches
    img_matches = cv2.drawMatches(uav_image, kp1, satellite_image, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # Save the image
    cv2.imwrite('feature_matches.jpg', img_matches)

if __name__ == '__main__':
    uav_image_path = 'data/example/images/rickmansworth_example.jpg'
    search_area_json_path = 'data/example/images/rickmansworth_example_search_area.json'
    visualize_feature_matching(uav_image_path, search_area_json_path)
