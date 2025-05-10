import sys
sys.path.append('/Users/pranavreddy/Documents/GitHub/25-LDTH-Relocalisation')
import json
import cv2
import numpy as np
from src.utility.get_satellite_image import get_satellite_image

def sift_baseline(uav_image_path, search_area_json_path):
    """
    A baseline for relocalisation that uses SIFT feature extraction and matching.
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

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(uav_image, None)
    kp2, des2 = sift.detectAndCompute(satellite_image, None)

    print(f"Number of UAV keypoints: {len(kp1)}")
    print(f"Number of satellite keypoints: {len(kp2)}")

    # Create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Extract matched keypoint locations
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Estimate homography matrix using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("Could not compute homography")
        predicted_lon = (min_lon + max_lon) / 2
        predicted_lat = (min_lat + max_lat) / 2
        return predicted_lon, predicted_lat

    # Use the homography matrix to transform the center of the UAV image
    h, w = uav_image.shape
    uav_center = np.float32([[[w/2, h/2]]])
    transformed_center = cv2.perspectiveTransform(uav_center, M)[0][0]

    # Convert the transformed coordinates to latitude and longitude
    satellite_h, satellite_w = satellite_image.shape
    predicted_x, predicted_y = transformed_center

    # Normalize x and y
    predicted_x /= satellite_w
    predicted_y /= satellite_h

    predicted_lon = min_lon + (max_lon - min_lon) * predicted_x
    predicted_lat = min_lat + (max_lat - min_lat) * predicted_y

    return predicted_lon, predicted_lat


if __name__ == '__main__':
    uav_image_path = 'data/example/images/rickmansworth_example.jpg'
    search_area_json_path = 'data/example/images/rickmansworth_example_search_area.json'
    predicted_lon, predicted_lat = sift_baseline(uav_image_path, search_area_json_path)
    print(f"Predicted Longitude: {predicted_lon}, Predicted Latitude: {predicted_lat}")
