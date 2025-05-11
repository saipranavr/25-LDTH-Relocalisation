import csv
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd # For reading the truth CSV easily

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Assuming helper scripts are callable and in the right paths
# We might need to import their functions directly if they are structured as modules,
# but for now, we'll plan to call them as subprocesses as we've been doing.

# --- Configuration ---
TRUTH_CSV_PATH = "data/example/images/test/test_images_truth.csv"
TEST_IMAGES_DIR = "data/example/images/test/"
MAIN_SEARCH_AREA_GEOJSON = "data/example/images/eiffel_tower_search_area.json" # Defines the overall region

# For Approach 1 (Direct Match)
DIRECT_SEARCH_IMG_SATELLITE = "data/example/images/eiffel_tower_search_area_satellite.jpg" # 1500x1500
DIRECT_SEARCH_IMG_WIDTH = 1500
DIRECT_SEARCH_IMG_HEIGHT = 1500
SUPERGLUE_OUTPUT_DIR_DIRECT = "data/example/superglue_batch_direct_output/"
TARGET_IMG_WIDTH = 512 # Assuming all test images are 512x512
TARGET_IMG_HEIGHT = 512

# For Approach 2 (Coarse-to-Fine) - details to be filled in
COARSE_SEARCH_IMG = "data/example/images/eiffel_tower_search_coarse.jpg" # 640x480
COARSE_SEARCH_IMG_WIDTH = 640
COARSE_SEARCH_IMG_HEIGHT = 480
DOWNSAMPLE_FACTOR = 4
TEMP_COARSE_TARGET_DIR = "data/example/images/temp_coarse_targets/"
SUPERGLUE_OUTPUT_DIR_COARSE = "data/example/superglue_batch_coarse_output/"
REFINED_SEARCH_AREA_SIZE_METERS = 1000 # e.g., 1km x 1km for refined search
REFINED_SEARCH_IMG_PIXELS = 1024
TEMP_REFINED_GEOJSON_DIR = "data/example/temp_refined_geojsons/"
TEMP_REFINED_SATELLITE_DIR = "data/example/images/temp_refined_satellite/"
SUPERGLUE_OUTPUT_DIR_REFINED = "data/example/superglue_batch_refined_output/"


RESULTS_CSV_PATH = "data/example/test_localization_results.csv"

# Ensure output directories exist
os.makedirs(SUPERGLUE_OUTPUT_DIR_DIRECT, exist_ok=True)
os.makedirs(TEMP_COARSE_TARGET_DIR, exist_ok=True)
os.makedirs(SUPERGLUE_OUTPUT_DIR_COARSE, exist_ok=True)
os.makedirs(TEMP_REFINED_GEOJSON_DIR, exist_ok=True)
os.makedirs(TEMP_REFINED_SATELLITE_DIR, exist_ok=True)
os.makedirs(SUPERGLUE_OUTPUT_DIR_REFINED, exist_ok=True)


def run_command(command_list):
    """Helper to run shell commands and print their output."""
    print(f"Executing: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        stdout, _ = process.communicate()
        print(stdout)
        if process.returncode != 0:
            print(f"Error: Command failed with return code {process.returncode}")
            return False, stdout
        return True, stdout
    except Exception as e:
        print(f"Exception running command: {e}")
        return False, str(e)

def parse_estimation_output(output_str):
    """Parses the output of estimate_location_from_matches.py to get lat, lon, and error."""
    est_lat, est_lon, error_m = None, None, None
    lines = output_str.splitlines()
    for i, line in enumerate(lines):
        # Check for final estimated coordinates
        if "Successfully estimated coordinates:" in line:
            if i + 1 < len(lines) and "Latitude (EPSG:4326):" in lines[i+1]:
                try:
                    est_lat = float(lines[i+1].split(":")[1].strip())
                except ValueError: pass
            if i + 2 < len(lines) and "Longitude (EPSG:4326):" in lines[i+2]:
                try:
                    est_lon = float(lines[i+2].split(":")[1].strip())
                except ValueError: pass
        
        # Check for error distance
        if "Error distance from truth" in line:
            if i + 1 < len(lines) and "meters" in lines[i+1]:
                try:
                    error_m = float(lines[i+1].strip().split(" ")[0])
                except ValueError: pass
                
    return est_lat, est_lon, error_m


def main():
    all_results = []
    
    try:
        truth_df = pd.read_csv(TRUTH_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Truth CSV not found at {TRUTH_CSV_PATH}")
        return

    # Pre-fetch large search area images if they don't exist (optional, assumes they are there for now)
    # For Approach 1:
    if not os.path.exists(DIRECT_SEARCH_IMG_SATELLITE):
        print(f"Fetching direct search area image: {DIRECT_SEARCH_IMG_SATELLITE}")
        # This would call src/fetch_search_area_image.py
        # python src/fetch_search_area_image.py MAIN_SEARCH_AREA_GEOJSON DIRECT_SEARCH_IMG_SATELLITE --width DIRECT_SEARCH_IMG_WIDTH --height DIRECT_SEARCH_IMG_HEIGHT
        # For simplicity, we assume it's pre-fetched or handle this manually first.
        # For now, let's assume it exists from previous steps.
        pass

    # For Approach 2 (Coarse step):
    if not os.path.exists(COARSE_SEARCH_IMG):
        print(f"Fetching coarse search area image: {COARSE_SEARCH_IMG}")
        # python src/fetch_search_area_image.py MAIN_SEARCH_AREA_GEOJSON COARSE_SEARCH_IMG --width COARSE_SEARCH_IMG_WIDTH --height COARSE_SEARCH_IMG_HEIGHT
        pass


    for index, row in truth_df.iterrows():
        image_id = row['image_id']
        true_lat = row['latitude']
        true_lon = row['longitude']
        
        target_image_path = os.path.join(TEST_IMAGES_DIR, image_id)
        print(f"\nProcessing {image_id} (True Lat: {true_lat:.6f}, True Lon: {true_lon:.6f})...")

        current_result = {
            "image_id": image_id,
            "true_lat": true_lat,
            "true_lon": true_lon,
            "approach1_est_lat": None, "approach1_est_lon": None, "approach1_error_m": None,
            "approach2_est_lat": None, "approach2_est_lon": None, "approach2_error_m": None,
        }

        # --- Approach 1: Direct High-Resolution Match ---
        print("\n--- Running Approach 1: Direct Match ---")
        
        # Ensure DIRECT_SEARCH_IMG_SATELLITE exists
        if not os.path.exists(DIRECT_SEARCH_IMG_SATELLITE):
            print(f"Error: Direct search image {DIRECT_SEARCH_IMG_SATELLITE} not found. Please generate it first.")
            # Optionally, add code here to call src/fetch_search_area_image.py
            # For now, we'll skip this image if the main search image is missing.
            all_results.append(current_result) # Add partial result
            continue

        # Create pair file with paths relative to project root
        # image_id is like "test_image_01.jpg"
        # target_image_path is "data/example/images/test/test_image_01.jpg"
        # DIRECT_SEARCH_IMG_SATELLITE is "data/example/images/eiffel_tower_search_area_satellite.jpg"
        
        # Paths for the pair file should be relative to the --input_dir for match_pairs.py
        # If --input_dir is '.', then these paths are correct as they are.
        path_img0_for_pairfile = target_image_path 
        path_img1_for_pairfile = DIRECT_SEARCH_IMG_SATELLITE

        pair_file_direct_name = f"{Path(image_id).stem}_direct_pair.txt"
        pair_file_direct_path = os.path.join(SUPERGLUE_OUTPUT_DIR_DIRECT, pair_file_direct_name)
        
        with open(pair_file_direct_path, 'w') as f:
            f.write(f"{path_img0_for_pairfile} {path_img1_for_pairfile}\n")

        # Define stems for output files based on the actual image filenames used in the pair file
        stem0_direct = Path(path_img0_for_pairfile).stem
        stem1_direct = Path(path_img1_for_pairfile).stem
        
        npz_direct_path = os.path.join(SUPERGLUE_OUTPUT_DIR_DIRECT, f"{stem0_direct}_{stem1_direct}_matches.npz")

        sg_cmd_direct = [
            "python", "third_party/SuperGluePretrainedNetwork/match_pairs.py",
            "--input_pairs", pair_file_direct_path,
            "--input_dir", ".", # Project root as input_dir
            "--output_dir", SUPERGLUE_OUTPUT_DIR_DIRECT,
            "--superglue", "outdoor", 
            "--viz", # Keep visualization for review if needed
            "--resize", "-1" # Crucial: no resize for direct high-res match
        ]
        
        success_sg_direct, _ = run_command(sg_cmd_direct)

        if success_sg_direct and os.path.exists(npz_direct_path):
            est_cmd_direct = [
                "python", "src/estimate_location_from_matches.py",
                npz_direct_path,
                MAIN_SEARCH_AREA_GEOJSON, # GeoJSON of the large search area image
                str(DIRECT_SEARCH_IMG_WIDTH), str(DIRECT_SEARCH_IMG_HEIGHT),
                str(TARGET_IMG_WIDTH), str(TARGET_IMG_HEIGHT),
                "--truth_lat", str(true_lat), "--truth_lon", str(true_lon)
            ]
            success_est_direct, out_direct = run_command(est_cmd_direct)
            if success_est_direct:
                lat, lon, err = parse_estimation_output(out_direct)
                current_result["approach1_est_lat"] = lat
                current_result["approach1_est_lon"] = lon
                current_result["approach1_error_m"] = err
        else:
            print(f"SuperGlue direct match failed or NPZ not found for {image_id}.")

        # --- Approach 2: Coarse-to-Fine Match ---
        print("\n--- Running Approach 2: Coarse-to-Fine Match ---")
        
        # Ensure COARSE_SEARCH_IMG exists
        if not os.path.exists(COARSE_SEARCH_IMG):
            print(f"Error: Coarse search image {COARSE_SEARCH_IMG} not found. Please generate it first.")
            all_results.append(current_result)
            continue # Skip to next image if coarse search base is missing

        # 2a. Downsample current target image
        coarse_target_image_name = f"{Path(image_id).stem}_coarse.jpg"
        coarse_target_image_path = os.path.join(TEMP_COARSE_TARGET_DIR, coarse_target_image_name)
        downsample_cmd = [
            "python", "src/downsample_image.py",
            target_image_path, coarse_target_image_path, str(DOWNSAMPLE_FACTOR)
        ]
        success_ds, _ = run_command(downsample_cmd)
        if not success_ds or not os.path.exists(coarse_target_image_path):
            print(f"Failed to downsample {target_image_path} for coarse match.")
            all_results.append(current_result)
            continue

        # 2b. Create pair file for coarse match (paths relative to project root for --input_dir .)
        path_img0_coarse_pairfile = coarse_target_image_path
        path_img1_coarse_pairfile = COARSE_SEARCH_IMG
        
        pair_file_coarse_name = f"{Path(coarse_target_image_name).stem}_{Path(COARSE_SEARCH_IMG).stem}_pair.txt"
        pair_file_coarse_path = os.path.join(SUPERGLUE_OUTPUT_DIR_COARSE, pair_file_coarse_name)
        with open(pair_file_coarse_path, 'w') as f:
            f.write(f"{path_img0_coarse_pairfile} {path_img1_coarse_pairfile}\n")

        stem0_coarse = Path(path_img0_coarse_pairfile).stem
        stem1_coarse = Path(path_img1_coarse_pairfile).stem
        npz_coarse_path = os.path.join(SUPERGLUE_OUTPUT_DIR_COARSE, f"{stem0_coarse}_{stem1_coarse}_matches.npz")

        sg_cmd_coarse = [
            "python", "third_party/SuperGluePretrainedNetwork/match_pairs.py",
            "--input_pairs", pair_file_coarse_path,
            "--input_dir", ".", # Project root
            "--output_dir", SUPERGLUE_OUTPUT_DIR_COARSE,
            "--superglue", "outdoor", "--viz" 
            # Default resize (640x480) is fine for coarse matching
        ]
        success_sg_coarse, _ = run_command(sg_cmd_coarse)

        if not success_sg_coarse or not os.path.exists(npz_coarse_path):
            print(f"SuperGlue coarse match failed or NPZ not found for {image_id}.")
            all_results.append(current_result)
            continue
            
        # 2c. Estimate coarse geographic center (EPSG:3035) from coarse matches
        # We need the *pixel* output from estimate_location_from_matches.py to guide refined search,
        # but also its EPSG:3035 estimate to center the new GeoJSON.
        est_cmd_coarse = [
            "python", "src/estimate_location_from_matches.py",
            npz_coarse_path,
            MAIN_SEARCH_AREA_GEOJSON, # GeoJSON of the COARSE_SEARCH_IMG
            str(COARSE_SEARCH_IMG_WIDTH), str(COARSE_SEARCH_IMG_HEIGHT),
            str(TARGET_IMG_WIDTH // DOWNSAMPLE_FACTOR), str(TARGET_IMG_HEIGHT // DOWNSAMPLE_FACTOR)
            # No truth lat/lon needed here, we need the estimated EPSG:3035 coords
        ]
        success_est_coarse, out_coarse = run_command(est_cmd_coarse)
        if not success_est_coarse:
            print(f"Coarse estimation failed for {image_id}.")
            all_results.append(current_result)
            continue

        # Parse output to get estimated EPSG:3035 coordinates
        # Example line: "Estimated EPSG:3035 Coords: (3756395.38, 2890309.13)"
        coarse_est_lon_3035, coarse_est_lat_3035 = None, None
        for line_out in out_coarse.splitlines():
            if "Estimated EPSG:3035 Coords:" in line_out:
                try:
                    # Content is like " (3756395.38, 2890309.13)"
                    coords_part = line_out.split(":", 1)[1].strip() # Get " (3756395.38, 2890309.13)"
                    coords_no_paren = coords_part.strip("()") # Get "3756395.38, 2890309.13"
                    lon_str, lat_str = coords_no_paren.split(",", 1) # Split only on the first comma
                    coarse_est_lon_3035 = float(lon_str.strip())
                    coarse_est_lat_3035 = float(lat_str.strip())
                    break
                except Exception as e:
                    print(f"Error parsing coarse EPSG:3035 coords: {e} from line: {line_out}")
        
        if coarse_est_lon_3035 is None or coarse_est_lat_3035 is None:
            print(f"Could not parse coarse EPSG:3035 coordinates for {image_id}.")
            all_results.append(current_result)
            continue
        print(f"Coarse estimated center (EPSG:3035): Lon={coarse_est_lon_3035}, Lat={coarse_est_lat_3035}")

        # 2d. Define refined search area bounds in EPSG:3035
        half_refined_size = REFINED_SEARCH_AREA_SIZE_METERS / 2
        refined_min_lon_3035 = coarse_est_lon_3035 - half_refined_size
        refined_max_lon_3035 = coarse_est_lon_3035 + half_refined_size
        refined_min_lat_3035 = coarse_est_lat_3035 - half_refined_size
        refined_max_lat_3035 = coarse_est_lat_3035 + half_refined_size

        # 2e. Create temporary GeoJSON for refined search area
        refined_geojson_filename = f"{Path(image_id).stem}_refined_search_area.json"
        refined_geojson_path = os.path.join(TEMP_REFINED_GEOJSON_DIR, refined_geojson_filename)
        create_geojson_cmd = [
            "python", "src/create_search_area_json.py",
            "--mode", "bounds_3035",
            "--min_lon_3035", str(refined_min_lon_3035),
            "--min_lat_3035", str(refined_min_lat_3035),
            "--max_lon_3035", str(refined_max_lon_3035),
            "--max_lat_3035", str(refined_max_lat_3035),
            refined_geojson_path
        ]
        success_create_geojson, _ = run_command(create_geojson_cmd)
        if not success_create_geojson or not os.path.exists(refined_geojson_path):
            print(f"Failed to create refined GeoJSON for {image_id}.")
            all_results.append(current_result)
            continue

        # 2f. Fetch high-res image for refined search area
        refined_satellite_image_name = f"{Path(image_id).stem}_refined_satellite.jpg"
        refined_satellite_image_path = os.path.join(TEMP_REFINED_SATELLITE_DIR, refined_satellite_image_name)
        fetch_refined_cmd = [
            "python", "src/fetch_search_area_image.py",
            refined_geojson_path, refined_satellite_image_path,
            "--width", str(REFINED_SEARCH_IMG_PIXELS), "--height", str(REFINED_SEARCH_IMG_PIXELS),
            "--no-display"
        ]
        success_fetch_refined, _ = run_command(fetch_refined_cmd)
        if not success_fetch_refined or not os.path.exists(refined_satellite_image_path):
            print(f"Failed to fetch refined satellite image for {image_id}.")
            all_results.append(current_result)
            continue
            
        # 2g. Create pair file for refined match (original target vs. new refined search image)
        path_img0_refined_pairfile = target_image_path # Original 512x512 target
        path_img1_refined_pairfile = refined_satellite_image_path
        
        pair_file_refined_name = f"{Path(image_id).stem}_refined_pair.txt"
        pair_file_refined_path = os.path.join(SUPERGLUE_OUTPUT_DIR_REFINED, pair_file_refined_name)
        with open(pair_file_refined_path, 'w') as f:
            f.write(f"{path_img0_refined_pairfile} {path_img1_refined_pairfile}\n")

        stem0_refined = Path(path_img0_refined_pairfile).stem
        stem1_refined = Path(path_img1_refined_pairfile).stem
        npz_refined_path = os.path.join(SUPERGLUE_OUTPUT_DIR_REFINED, f"{stem0_refined}_{stem1_refined}_matches.npz")

        sg_cmd_refined = [
            "python", "third_party/SuperGluePretrainedNetwork/match_pairs.py",
            "--input_pairs", pair_file_refined_path,
            "--input_dir", ".", # Project root
            "--output_dir", SUPERGLUE_OUTPUT_DIR_REFINED,
            "--superglue", "outdoor", "--viz", "--resize", "-1"
        ]
        success_sg_refined, _ = run_command(sg_cmd_refined)

        if not success_sg_refined or not os.path.exists(npz_refined_path):
            print(f"SuperGlue refined match failed or NPZ not found for {image_id}.")
            all_results.append(current_result)
            continue

        # 2h. Estimate final location from refined matches
        est_cmd_refined = [
            "python", "src/estimate_location_from_matches.py",
            npz_refined_path,
            refined_geojson_path, # GeoJSON of the refined search area image
            str(REFINED_SEARCH_IMG_PIXELS), str(REFINED_SEARCH_IMG_PIXELS),
            str(TARGET_IMG_WIDTH), str(TARGET_IMG_HEIGHT),
            "--truth_lat", str(true_lat), "--truth_lon", str(true_lon)
        ]
        success_est_refined, out_refined = run_command(est_cmd_refined)
        if success_est_refined:
            lat_ref, lon_ref, err_ref = parse_estimation_output(out_refined)
            current_result["approach2_est_lat"] = lat_ref
            current_result["approach2_est_lon"] = lon_ref
            current_result["approach2_error_m"] = err_ref
        else:
            print(f"Refined estimation failed for {image_id}.")

        all_results.append(current_result)

    # Write all results to CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(RESULTS_CSV_PATH, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nBatch localization complete. Results saved to {RESULTS_CSV_PATH}")
    else:
        print("No results to save.")

if __name__ == '__main__':
    main()
