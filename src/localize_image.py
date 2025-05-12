import math
import argparse
import csv
import os
# import cv2 # Placeholder for OpenCV, if needed for image processing
# import numpy as np # Placeholder for NumPy, if needed

# Fixed zoom level
ZOOM_LEVEL = 19

# Placeholder for a function to get satellite image for a tile
# This would likely call an external API or use a library
def get_satellite_image_for_tile(tile_x, tile_y, zoom, output_dir="temp_satellite_tiles"):
    """
    Placeholder: Fetches or generates a satellite image for the given tile.
    Returns the path to the fetched image.
    """
    os.makedirs(output_dir, exist_ok=True)
    # This is a dummy implementation. In a real scenario, you'd fetch an image.
    # For example, using a function from `src/utility/get_satellite_image.py`
    dummy_image_path = os.path.join(output_dir, f"tile_{zoom}_{tile_x}_{tile_y}.png")
    # Create a dummy file to simulate a downloaded image
    with open(dummy_image_path, 'w') as f:
        f.write("dummy satellite image content")
    print(f"Placeholder: Fetched satellite image for tile ({tile_x}, {tile_y}, {zoom}) to {dummy_image_path}")
    return dummy_image_path

# Placeholder for feature extraction
def extract_features(image_path):
    """
    Placeholder: Extracts features from an image.
    Returns a representation of the features.
    """
    # In a real implementation, this would use a library like OpenCV, SuperPoint, etc.
    # features = cv2.SIFT_create().detectAndCompute(cv2.imread(image_path), None)
    print(f"Placeholder: Extracted features from {image_path}")
    return {"path": image_path, "data": "dummy_features"} # Dummy features

# Placeholder for feature matching (ANN lookup)
def find_best_match(uav_features, satellite_features_list):
    """
    Placeholder: Finds the best match for UAV features among a list of satellite features.
    Returns the index of the best matching satellite image and match quality.
    """
    # In a real implementation, this would use ANN (e.g., FAISS) or robust matchers (e.g., SuperGlue)
    print(f"Placeholder: Comparing UAV features with {len(satellite_features_list)} satellite images.")
    if not satellite_features_list:
        return None, 0
    # Dummy matching: assume the first one is the best for now
    best_match_index = 0
    match_quality = 0.9 # Dummy quality
    print(f"Placeholder: Best match found with satellite image index {best_match_index} (quality: {match_quality})")
    return best_match_index, match_quality

def latlon_to_tile(lat, lon, z):
    """
    Converts latitude and longitude to Web Mercator tile indices.
    """
    n = 2 ** z
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(math.radians(lat)) +
              1/math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return int(xtile), int(ytile)

def enumerate_sub_tiles(lat_min, lon_min, lat_max, lon_max, zoom):
    """
    Enumerates all tiles within the given bounding box at the specified zoom level.
    It uses the tile conversion logic as specified in the problem description:
    x_min_tile, y_max_tile = latlon_to_tile(lat_min, lon_min, zoom)
    x_max_tile, y_min_tile = latlon_to_tile(lat_max, lon_max, zoom)
    The iteration then proceeds over the rectangle defined by these tile indices.
    """
    # Convert the two corner points of the AOI to tile indices
    # Corner 1 (e.g., top-left or bottom-left, depending on user's lat_min/lon_min definition)
    tile_x1, tile_y1 = latlon_to_tile(lat_min, lon_min, zoom)
    # Corner 2 (e.g., bottom-right or top-right)
    tile_x2, tile_y2 = latlon_to_tile(lat_max, lon_max, zoom)

    # Determine the iteration range for x and y tile coordinates
    # x_start should be the minimum of the two x tile coordinates
    # x_end should be the maximum of the two x tile coordinates
    # y_start should be the minimum of the two y tile coordinates (tiles indexed from top)
    # y_end should be the maximum of the two y tile coordinates
    
    x_start = min(tile_x1, tile_x2)
    x_end = max(tile_x1, tile_x2)
    y_start = min(tile_y1, tile_y2) # y tile indices increase downwards (north to south)
    y_end = max(tile_y1, tile_y2)

    tiles = []
    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
            tiles.append((x, y, zoom))
    return tiles

def tile_to_latlon(xtile, ytile, zoom):
    """
    Converts Web Mercator tile coordinates (x, y, zoom) back to latitude and longitude (center of the tile).
    """
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def search_in_area(input_image_path, sub_tiles):
    """
    Searches for the input image within the given sub-tiles using ANN lookup.
    """
    print(f"Starting search for UAV image: {input_image_path}")

    if not os.path.exists(input_image_path):
        print(f"Error: Input UAV image not found at {input_image_path}")
        return None, None

    # 1. Extract features from the input UAV image
    uav_features = extract_features(input_image_path)
    if not uav_features:
        print(f"Error: Could not extract features from UAV image {input_image_path}")
        return None, None

    satellite_image_features_list = []
    processed_satellite_tiles = []

    # 2. For each sub-tile:
    for i, (tile_x, tile_y, tile_z) in enumerate(sub_tiles):
        print(f"Processing sub-tile {i+1}/{len(sub_tiles)}: ({tile_x}, {tile_y}, {tile_z})")
        #   a. Fetch/generate the satellite image for the sub-tile
        satellite_image_path = get_satellite_image_for_tile(tile_x, tile_y, tile_z)
        if not satellite_image_path or not os.path.exists(satellite_image_path):
            print(f"Warning: Could not get satellite image for tile ({tile_x}, {tile_y}, {tile_z}). Skipping.")
            continue
        
        #   b. Extract features from the satellite sub-tile image
        sat_features = extract_features(satellite_image_path)
        if sat_features:
            satellite_image_features_list.append(sat_features)
            processed_satellite_tiles.append((tile_x, tile_y, tile_z))
        else:
            print(f"Warning: Could not extract features from satellite image {satellite_image_path}. Skipping.")
        
        # Optional: Clean up temporary satellite image if it's large or many
        # os.remove(satellite_image_path)


    if not satellite_image_features_list:
        print("No satellite images could be processed or features extracted.")
        return None, None

    # 3. Perform matching (ANN lookup) to find the best satellite tile
    best_match_index, match_quality = find_best_match(uav_features, satellite_image_features_list)

    if best_match_index is not None and match_quality > 0.5: # Assuming 0.5 is a threshold
        best_matching_tile = processed_satellite_tiles[best_match_index]
        tile_x, tile_y, tile_z = best_matching_tile
        
        # 4. If a good match is found, estimate lat/lon (e.g., center of the best matching tile)
        # More sophisticated estimation could involve homography or geometric verification
        est_lat, est_lon = tile_to_latlon(tile_x + 0.5, tile_y + 0.5, tile_z) # Center of the tile
        print(f"Best match found with tile {best_matching_tile} (Quality: {match_quality}). Estimated Lat/Lon: ({est_lat}, {est_lon})")
        return est_lat, est_lon
    else:
        print("No confident match found after searching all sub-tiles.")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Localize an image within a given lat/lon search space.")
    parser.add_argument("input_image", help="Path to the input image file (e.g., data/example/images/test/test_image_01.jpg).")
    parser.add_argument("lat_min", type=float, help="Minimum latitude of the search area (southern boundary).")
    parser.add_argument("lon_min", type=float, help="Minimum longitude of the search area (western boundary).")
    parser.add_argument("lat_max", type=float, help="Maximum latitude of the search area (northern boundary).")
    parser.add_argument("lon_max", type=float, help="Maximum longitude of the search area (eastern boundary).")
    parser.add_argument("--output_csv", default="estimations_localize_image.csv", help="Path to save the output CSV file.")

    args = parser.parse_args()

    # Extract image_id from the input_image path
    image_id = os.path.basename(args.input_image)

    print(f"Input image ID: {image_id}")
    print(f"Input image path: {args.input_image}")
    print(f"Search Area: Lat ({args.lat_min}, {args.lat_max}), Lon ({args.lon_min}, {args.lon_max})")
    print(f"Zoom level: {ZOOM_LEVEL}")

    # Convert AOI to tile bounds and enumerate sub-tiles
    # Note: The problem description's variable names for latlon_to_tile output were a bit confusing.
    # lat_min, lon_min should be bottom-left of AOI
    # lat_max, lon_max should be top-right of AOI
    # However, the example `x_min, y_max = latlon_to_tile(lat_min, lon_min, Z)` implies lat_min is top.
    # Let's assume standard geographic coordinates:
    # lat_min = southern boundary, lat_max = northern boundary
    # lon_min = western boundary, lon_max = eastern boundary

    # Top-left corner of the AOI for tiling: (lat_max, lon_min)
    # Bottom-right corner of the AOI for tiling: (lat_min, lon_max)

    # Tile containing the top-left point of the AOI
    # x_start_tile, y_start_tile = latlon_to_tile(args.lat_max, args.lon_min, ZOOM_LEVEL)
    # Tile containing the bottom-right point of the AOI
    # x_end_tile, y_end_tile = latlon_to_tile(args.lat_min, args.lon_max, ZOOM_LEVEL)

    # print(f"Calculated tile range: X ({x_start_tile}-{x_end_tile}), Y ({y_start_tile}-{y_end_tile})")

    sub_tiles = enumerate_sub_tiles(args.lat_min, args.lon_min, args.lat_max, args.lon_max, ZOOM_LEVEL)

    if not sub_tiles:
        print("No sub-tiles generated for the given AOI. Check coordinates and zoom level.")
        return

    print(f"Generated {len(sub_tiles)} sub-tiles to search.")
    # for tile in sub_tiles:
    #     print(f"  Tile: {tile}") # Can be verbose for many tiles

    # Perform search
    estimated_lat, estimated_lon = search_in_area(args.input_image, sub_tiles)

    # Prepare header for CSV
    file_exists = os.path.isfile(args.output_csv)
    
    with open(args.output_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists or os.path.getsize(args.output_csv) == 0:
            csv_writer.writerow(["image_id", "latitude", "longitude"]) # Header

        if estimated_lat is not None and estimated_lon is not None:
            print(f"Final Estimated location for {image_id}: Latitude={estimated_lat}, Longitude={estimated_lon}")
            csv_writer.writerow([image_id, estimated_lat, estimated_lon])
        else:
            print(f"Could not estimate location for {image_id}. Writing N/A to CSV.")
            csv_writer.writerow([image_id, "N/A", "N/A"])
    
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
