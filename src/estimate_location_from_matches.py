import numpy as np
import cv2
import argparse
import json
from pyproj import Transformer
from pathlib import Path
import sys
import math # For Haversine distance

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utility.epsg_4326_to_3035 import geojson_to_3035_bboxes
from src.utility.bounding_box import Projection, BoundingBox # For type hinting and Projection enum

def estimate_lat_lon_from_matches(
    npz_path: str, 
    search_area_geojson_path: str, 
    search_area_img_width: int, 
    search_area_img_height: int,
    target_img_width: int, # Used to find center of target image
    target_img_height: int # Used to find center of target image
    ):
    """
    Estimates the lat/lon of a target image within a larger search area image
    using SuperGlue match outputs.
    """
    print(f"Loading matches from: {npz_path}")
    try:
        match_data = np.load(npz_path)
    except Exception as e:
        print(f"Error loading match data: {e}")
        return None

    mkpts0 = match_data.get('keypoints0')[match_data.get('matches') > -1] # Keypoints from target image
    mkpts1 = match_data.get('keypoints1')[match_data.get('matches')[match_data.get('matches') > -1]] # Corresponding keypoints in search area image
    
    if len(mkpts0) < 4: # Need at least 4 points for homography
        print(f"Error: Not enough matches ({len(mkpts0)}) to estimate homography.")
        return None
    
    print(f"Found {len(mkpts0)} matches.")

    # 1. Estimate Homography (target image to search area image)
    try:
        homography, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
        if homography is None:
            print("Error: Homography estimation failed.")
            return None
        num_inliers = np.sum(mask)
        print(f"Homography estimated with {num_inliers} inliers out of {len(mkpts0)} matches.")
        if num_inliers < 4: # Arbitrary threshold, might need tuning
             print(f"Warning: Low number of inliers ({num_inliers}) for homography. Result might be unreliable.")

    except cv2.error as e:
        print(f"OpenCV error during findHomography: {e}")
        return None


    # 2. Define reference point in target image (its center)
    target_center_pt = np.array([[[target_img_width / 2, target_img_height / 2]]], dtype=np.float32)

    # 3. Transform target center to search area image coordinates
    transformed_center_pt = cv2.perspectiveTransform(target_center_pt, homography)
    if transformed_center_pt is None or transformed_center_pt.shape[1] == 0:
        print("Error: Perspective transform failed or resulted in no points.")
        return None
        
    search_area_pixel_x = transformed_center_pt[0][0][0]
    search_area_pixel_y = transformed_center_pt[0][0][1]
    print(f"Target center transformed to pixel coords in search area image: ({search_area_pixel_x:.2f}, {search_area_pixel_y:.2f})")

    # 4. Convert search area pixel coordinates to geographic coordinates
    # 4a. Get EPSG:3035 bounding box of the search area
    print(f"Loading search area GeoJSON: {search_area_geojson_path}")
    search_area_bboxes_3035 = geojson_to_3035_bboxes(search_area_geojson_path)
    if not search_area_bboxes_3035:
        print(f"Error: Could not derive EPSG:3035 bounding box from {search_area_geojson_path}")
        return None
    sa_bbox_3035: BoundingBox = search_area_bboxes_3035[0]
    
    min_lon_3035, max_lon_3035 = sa_bbox_3035.min_lon, sa_bbox_3035.max_lon
    min_lat_3035, max_lat_3035 = sa_bbox_3035.min_lat, sa_bbox_3035.max_lat
    print(f"Search Area EPSG:3035 BBox: Lon({min_lon_3035:.2f} to {max_lon_3035:.2f}), Lat({min_lat_3035:.2f} to {max_lat_3035:.2f})")

    # 4b. Calculate meters per pixel
    lon_range_meters = max_lon_3035 - min_lon_3035
    lat_range_meters = max_lat_3035 - min_lat_3035
    
    meters_per_pixel_lon = lon_range_meters / search_area_img_width
    meters_per_pixel_lat = lat_range_meters / search_area_img_height
    print(f"Meters per pixel: Lon={meters_per_pixel_lon:.4f}, Lat={meters_per_pixel_lat:.4f}")

    # 4c. Calculate EPSG:3035 coordinates of the transformed point
    # Note: Image Y origin is top-left, geographic Y (latitude) origin is bottom-left.
    est_lon_3035 = min_lon_3035 + (search_area_pixel_x * meters_per_pixel_lon)
    est_lat_3035 = max_lat_3035 - (search_area_pixel_y * meters_per_pixel_lat) # Y is inverted
    print(f"Estimated EPSG:3035 Coords: ({est_lon_3035:.2f}, {est_lat_3035:.2f})")

    # 5. Convert estimated EPSG:3035 coordinates to EPSG:4326 (lat/lon)
    transformer_3035_to_4326 = Transformer.from_crs(Projection.EPSG_3035.value, Projection.EPSG_4326.value, always_xy=True)
    est_lon_4326, est_lat_4326 = transformer_3035_to_4326.transform(est_lon_3035, est_lat_3035)
    
    print(f"Estimated EPSG:4326 Coords (Lon, Lat): ({est_lon_4326:.6f}, {est_lat_4326:.6f})")
    return est_lat_4326, est_lon_4326

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Radius of Earth in meters
    return c * r

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate Lat/Lon from SuperGlue matches and optionally calculate error distance.")
    parser.add_argument("npz_file", help="Path to the SuperGlue .npz matches file.")
    parser.add_argument("search_area_geojson", help="Path to the GeoJSON file of the search area.")
    parser.add_argument("search_img_width", type=int, help="Width of the search area satellite image in pixels.")
    parser.add_argument("search_img_height", type=int, help="Height of the search area satellite image in pixels.")
    parser.add_argument("target_img_width", type=int, help="Width of the target image in pixels (e.g., 512).")
    parser.add_argument("target_img_height", type=int, help="Height of the target image in pixels (e.g., 512).")
    parser.add_argument("--truth_lat", type=float, help="Ground truth latitude (EPSG:4326) for error calculation.")
    parser.add_argument("--truth_lon", type=float, help="Ground truth longitude (EPSG:4326) for error calculation.")

    args = parser.parse_args()

    estimated_coords = estimate_lat_lon_from_matches(
        args.npz_file,
        args.search_area_geojson,
        args.search_img_width,
        args.search_img_height,
        args.target_img_width,
        args.target_img_height
    )

    if estimated_coords:
        est_lat, est_lon = estimated_coords
        print(f"\nSuccessfully estimated coordinates:")
        print(f"  Latitude (EPSG:4326): {est_lat:.6f}")
        print(f"  Longitude (EPSG:4326): {est_lon:.6f}")

        if args.truth_lat is not None and args.truth_lon is not None:
            error_dist = haversine_distance(args.truth_lat, args.truth_lon, est_lat, est_lon)
            print(f"\nError distance from truth ({args.truth_lat:.6f}, {args.truth_lon:.6f}):")
            print(f"  {error_dist:.2f} meters")
    else:
        print("\nFailed to estimate coordinates.")
