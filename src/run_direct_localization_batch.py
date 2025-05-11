import csv
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import argparse
import re
import json # Added for loading tile metadata and creating temp geojsons

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utility.create_staticmap_tiles import (
    generate_tiles_for_geojson,
    DEFAULT_ZOOM_LEVEL as DEFAULT_TILE_ZOOM,
    DEFAULT_TILE_WIDTH_PX as DEFAULT_TILE_WIDTH,
    DEFAULT_TILE_HEIGHT_PX as DEFAULT_TILE_HEIGHT
)

# Configuration for paths
DEFAULT_SUPERGLUE_SCRIPT_PATH = "third_party/SuperGluePretrainedNetwork/match_pairs.py"
DEFAULT_ESTIMATOR_SCRIPT_PATH = "src/estimate_location_from_matches.py"
# DEFAULT_FETCHER_SCRIPT_PATH removed as tiling script handles fetching
DEFAULT_SUPERGLUE_OUTPUT_DIR = "data/example/direct_localization_output_tiled/superglue/" # Updated output dir
DEFAULT_TILES_OUTPUT_DIR = "data/example/direct_localization_output_tiled/generated_tiles/" # For generated tiles
DEFAULT_TARGET_IMG_WIDTH = 512 # Pixel width of the small target images
DEFAULT_TARGET_IMG_HEIGHT = 512 # Pixel height of the small target images

def run_command(command_list):
    """Helper to run shell commands and print their output."""
    # print(f"Executing: {' '.join(command_list)}") # DEBUG
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        stdout, _ = process.communicate()
        # print(stdout) # DEBUG: Potentially very verbose
        if process.returncode != 0:
            # print(f"Error: Command failed with return code {process.returncode}") # DEBUG
            # print(f"Output: {stdout}") # DEBUG
            return False, stdout
        return True, stdout
    except Exception as e:
        # print(f"Exception running command: {e}") # DEBUG
        return False, str(e)

# display_images_for_matching function removed as it's not suitable for batch processing many tiles.

def parse_estimation_output_for_coords(output_str):
    """Parses the output of estimate_location_from_matches.py to get only lat, lon."""
    est_lat, est_lon = None, None
    lines = output_str.splitlines()
    found_success_header = False

    for line in lines:
        line = line.strip()

        if "Successfully estimated coordinates:" in line:
            found_success_header = True
            continue

        if found_success_header:
            # Match the last float number in the line
            float_match = re.search(r"([-+]?[0-9]*\.?[0-9]+)$", line)

            if "Latitude" in line and float_match:
                est_lat = float(float_match.group(1))

            elif "Longitude" in line and float_match:
                est_lon = float(float_match.group(1))
            if est_lat is not None and est_lon is not None:
                break
    return est_lat, est_lon

def main():
    parser = argparse.ArgumentParser(description="Run direct localization for a batch of images using tiled satellite imagery.")
    parser.add_argument("truth_csv", help="Path to the input truth CSV (image_id,latitude,longitude).")
    parser.add_argument("target_images_base_dir", help="Base directory where target images (from truth_csv) are located.")
    parser.add_argument("search_area_geojson", help="Path to the GeoJSON defining the overall search area's bounds.")
    parser.add_argument("output_estimations_csv", help="Path to save the output estimations CSV.")
    
    # Tile generation parameters
    parser.add_argument("--tile_output_dir", default=DEFAULT_TILES_OUTPUT_DIR, help="Directory to save generated satellite tiles and metadata.")
    parser.add_argument("--tile_zoom", type=int, default=DEFAULT_TILE_ZOOM, help="Zoom level for satellite tiles.")
    parser.add_argument("--tile_width", type=int, default=DEFAULT_TILE_WIDTH, help="Width of each satellite tile in pixels.")
    parser.add_argument("--tile_height", type=int, default=DEFAULT_TILE_HEIGHT, help="Height of each satellite tile in pixels.")
    parser.add_argument("--tile_overlap", type=float, default=0.1, help="Overlap percentage between tiles (0.0 to 0.9).")

    args = parser.parse_args()

    os.makedirs(DEFAULT_SUPERGLUE_OUTPUT_DIR, exist_ok=True)
    # tile_output_dir will be created by generate_tiles_for_geojson if it doesn't exist
    
    # Directory for temporary GeoJSON files for each tile
    temp_tile_geojson_dir = Path(args.tile_output_dir) / "temp_tile_geojsons"
    os.makedirs(temp_tile_geojson_dir, exist_ok=True)

    all_estimations = []

    try:
        truth_df = pd.read_csv(args.truth_csv)
    except FileNotFoundError:
        print(f"Error: Truth CSV not found at {args.truth_csv}")
        return

    if not os.path.exists(args.search_area_geojson):
        print(f"Error: Search area GeoJSON not found at {args.search_area_geojson}")
        return

    # 1. Generate/Fetch all satellite tiles for the search area
    print(f"Generating satellite tiles for {args.search_area_geojson}...")
    tile_gen_result = generate_tiles_for_geojson(
        geojson_path=args.search_area_geojson,
        output_base_dir=args.tile_output_dir,
        tile_zoom=args.tile_zoom,
        tile_pixels_w=args.tile_width,
        tile_pixels_h=args.tile_height,
        overlap_percentage=args.tile_overlap
    )

    if "error" in tile_gen_result or tile_gen_result.get("tile_count", 0) == 0:
        print(f"Failed to generate tiles or no tiles generated: {tile_gen_result.get('error', 'No tiles')}")
        return
    
    print(f"Successfully generated {tile_gen_result['tile_count']} tiles. Metadata: {tile_gen_result['metadata_path']}")
    
    with open(tile_gen_result['metadata_path'], 'r') as f:
        tiles_metadata_collection = json.load(f)
    
    satellite_tiles_info = tiles_metadata_collection.get("tiles", [])
    if not satellite_tiles_info:
        print("No tile information found in metadata. Exiting.")
        return

    for index, row in truth_df.iterrows():
        image_id_filename = row['image_id'] # e.g., "test_image_01.jpg"
        target_image_full_path = os.path.join(args.target_images_base_dir, image_id_filename)

        if not os.path.exists(target_image_full_path):
            print(f"Warning: Target image {target_image_full_path} not found. Skipping {image_id_filename}.")
            all_estimations.append({"image_id": image_id_filename, "latitude": "", "longitude": ""})
            continue
            
        print(f"\nProcessing Target Image: {image_id_filename}...")
        
        est_lat, est_lon = None, None
        match_found_for_target = False

        for tile_info in satellite_tiles_info:
            if match_found_for_target: # If we already found a match for this target, skip other tiles
                break

            tile_id = tile_info['tile_id']
            satellite_tile_path = tile_info['absolute_path']
            # print(f"  Attempting match with Tile ID: {tile_id} ({Path(satellite_tile_path).name})") # DEBUG

            if not os.path.exists(satellite_tile_path):
                # print(f"    Warning: Satellite tile image {satellite_tile_path} not found. Skipping tile.") # DEBUG
                continue

            # Create a temporary GeoJSON for this specific tile's bounds
            current_tile_geojson_content = {
                "type": "FeatureCollection",
                "features": [tile_info['geojson_feature']]
            }
            temp_geojson_filename = f"tile_{tile_id}_bounds.geojson"
            current_tile_geojson_path = temp_tile_geojson_dir / temp_geojson_filename
            with open(current_tile_geojson_path, 'w') as f_geojson:
                json.dump(current_tile_geojson_content, f_geojson)

            # Create pair file for SuperGlue
            pair_file_name = f"{Path(image_id_filename).stem}_vs_tile{tile_id}_pair.txt"
            pair_file_path = os.path.join(DEFAULT_SUPERGLUE_OUTPUT_DIR, pair_file_name)
            with open(pair_file_path, 'w') as f:
                f.write(f"{target_image_full_path} {satellite_tile_path}\n")

            # Define stems for SuperGlue output NPZ file
            stem0 = Path(target_image_full_path).stem
            stem1 = f"tile{tile_id}" # Use tile_id for uniqueness
            npz_matches_path = os.path.join(DEFAULT_SUPERGLUE_OUTPUT_DIR, f"{stem0}_{stem1}_matches.npz")
            
            # Run SuperGlue
            sg_cmd = [
                "python", DEFAULT_SUPERGLUE_SCRIPT_PATH,
                "--input_pairs", str(pair_file_path),
                "--input_dir", str(project_root), 
                "--output_dir", DEFAULT_SUPERGLUE_OUTPUT_DIR,
                "--superglue", "outdoor",
                "--resize", "-1" 
            ]
            success_sg, sg_out = run_command(sg_cmd)

            if success_sg and os.path.exists(npz_matches_path):
                est_cmd = [
                    "python", DEFAULT_ESTIMATOR_SCRIPT_PATH,
                    npz_matches_path,
                    str(current_tile_geojson_path), 
                    str(tile_info['width_px']), str(tile_info['height_px']), 
                    str(DEFAULT_TARGET_IMG_WIDTH), str(DEFAULT_TARGET_IMG_HEIGHT)
                ]
                success_est, out_est = run_command(est_cmd)
                if success_est:
                    est_lat_tile, est_lon_tile = parse_estimation_output_for_coords(out_est)
                    if est_lat_tile is not None and est_lon_tile is not None:
                        print(f"  SUCCESS: Match found for {image_id_filename} with Tile ID {tile_id}. Lat: {est_lat_tile}, Lon: {est_lon_tile}")
                        est_lat, est_lon = est_lat_tile, est_lon_tile
                        match_found_for_target = True
                    # else: # DEBUG
                        # print(f"    Estimation parsing failed for Tile ID {tile_id}. Output: {out_est[:200]}") # DEBUG
                # else: # DEBUG
                    # print(f"    Estimation command failed for Tile ID {tile_id}. Output: {out_est[:200]}") # DEBUG
            else: 
                print(f"    SuperGlue match failed or NPZ not found for {image_id_filename} vs Tile {tile_id}.") # DEBUG ENABLED
                if not success_sg: print(f"    SuperGlue Output: {sg_out[:500]}") # DEBUG ENABLED
        
        if not match_found_for_target:
            print(f"  No successful match found for {image_id_filename} against any of the {len(satellite_tiles_info)} tiles.") # DEBUG ENABLED

        all_estimations.append({"image_id": image_id_filename, "latitude": est_lat, "longitude": est_lon})

    # Clean up temporary GeoJSON directory (optional)
    # import shutil
    # if temp_tile_geojson_dir.exists():
    #     print(f"Cleaning up temporary tile GeoJSONs in {temp_tile_geojson_dir}...")
    #     shutil.rmtree(temp_tile_geojson_dir)
    #     print("Cleanup complete.")

    if all_estimations:
        fieldnames = ["image_id", "latitude", "longitude"]
        with open(args.output_estimations_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row_data in all_estimations:
                row_data_formatted = {
                    "image_id": row_data["image_id"],
                    "latitude": f"{row_data['latitude']:.7f}" if row_data['latitude'] is not None else "",
                    "longitude": f"{row_data['longitude']:.7f}" if row_data['longitude'] is not None else ""
                }
                writer.writerow(row_data_formatted)
        print(f"\nBatch direct localization complete. Estimations saved to {args.output_estimations_csv}")
    else:
        print("No estimations to save.")

if __name__ == '__main__':
    main()
