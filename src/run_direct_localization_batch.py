import csv
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import argparse
import re
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Configuration for paths that might be relatively fixed for this specific task
DEFAULT_SUPERGLUE_SCRIPT_PATH = "third_party/SuperGluePretrainedNetwork/match_pairs.py"
DEFAULT_ESTIMATOR_SCRIPT_PATH = "src/estimate_location_from_matches.py"
DEFAULT_FETCHER_SCRIPT_PATH = "src/fetch_search_area_image.py" # Added
DEFAULT_SUPERGLUE_OUTPUT_DIR = "data/example/direct_localization_output/superglue/"
DEFAULT_TARGET_IMG_WIDTH = 512 # Pixel width of the small target images
DEFAULT_TARGET_IMG_HEIGHT = 512 # Pixel height of the small target images

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

def display_images_for_matching(target_img_path: str, satellite_img_path: str):
    """Displays the target and satellite images using Matplotlib."""
    try:
        target_img = Image.open(target_img_path)
        satellite_img = Image.open(satellite_img_path)

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        axes[0].imshow(target_img)
        axes[0].set_title(f"Target: {os.path.basename(target_img_path)}")
        axes[0].axis('off')
        
        axes[1].imshow(satellite_img)
        axes[1].set_title(f"Satellite Search Area: {os.path.basename(satellite_img_path)}")
        axes[1].axis('off')
        
        plt.suptitle("Images for SuperGlue Matching")
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plt.show() # This will block until the window is closed

    except FileNotFoundError:
        print(f"Error displaying images: One or both files not found ({target_img_path}, {satellite_img_path})")
    except Exception as e:
        print(f"Error displaying images: {e}")

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
    parser = argparse.ArgumentParser(description="Run direct localization for a batch of images.")
    parser.add_argument("truth_csv", help="Path to the input truth CSV (image_id,latitude,longitude).")
    parser.add_argument("target_images_base_dir", help="Base directory where target images (from truth_csv) are located.")
    # Removed search_area_image, will be derived from geojson
    parser.add_argument("search_area_geojson", help="Path to the GeoJSON defining the search area's bounds.")
    parser.add_argument("search_area_img_width", type=int, help="Desired width of the search area image in pixels.")
    parser.add_argument("search_area_img_height", type=int, help="Desired height of the search area image in pixels.")
    parser.add_argument("output_estimations_csv", help="Path to save the output estimations CSV.")
    parser.add_argument("--search_area_image_output_dir", default="data/example/images/", help="Directory to save/find the fetched large search area image.")
    
    args = parser.parse_args()

    os.makedirs(DEFAULT_SUPERGLUE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(args.search_area_image_output_dir, exist_ok=True)
    
    all_estimations = []

    try:
        truth_df = pd.read_csv(args.truth_csv)
    except FileNotFoundError:
        print(f"Error: Truth CSV not found at {args.truth_csv}")
        return

    if not os.path.exists(args.search_area_geojson):
        print(f"Error: Search area GeoJSON not found at {args.search_area_geojson}")
        return

    # Define and fetch/check the main search area satellite image
    search_area_image_filename = f"{Path(args.search_area_geojson).stem}_satellite_{args.search_area_img_width}x{args.search_area_img_height}.jpg"
    search_area_image_path = os.path.join(args.search_area_image_output_dir, search_area_image_filename)

    if not os.path.exists(search_area_image_path):
        print(f"Search area image {search_area_image_path} not found. Attempting to fetch...")
        fetch_cmd = [
            "python", DEFAULT_FETCHER_SCRIPT_PATH,
            args.search_area_geojson,
            search_area_image_path,
            "--width", str(args.search_area_img_width),
            "--height", str(args.search_area_img_height),
            "--no-display"
        ]
        success_fetch, _ = run_command(fetch_cmd)
        if not success_fetch or not os.path.exists(search_area_image_path):
            print(f"Failed to fetch search area image. Exiting.")
            return
    else:
        print(f"Using existing search area image: {search_area_image_path}")


    for index, row in truth_df.iterrows():
        image_id_filename = row['image_id'] # e.g., "test_image_01.jpg"
        
        target_image_full_path = os.path.join(args.target_images_base_dir, image_id_filename)

        if not os.path.exists(target_image_full_path):
            print(f"Warning: Target image {target_image_full_path} not found. Skipping.")
            all_estimations.append({"image_id": image_id_filename, "latitude": "", "longitude": ""})
            continue
            
        print(f"\nProcessing {image_id_filename}...")

        # Create pair file for SuperGlue
        path_img0_for_pairfile = target_image_full_path 
        path_img1_for_pairfile = search_area_image_path # Use the fetched/checked path

        pair_file_name = f"{Path(image_id_filename).stem}_vs_{Path(search_area_image_path).stem}_pair.txt"
        pair_file_path = os.path.join(DEFAULT_SUPERGLUE_OUTPUT_DIR, pair_file_name)
        
        with open(pair_file_path, 'w') as f:
            f.write(f"{path_img0_for_pairfile} {path_img1_for_pairfile}\n")

        # Define stems for SuperGlue output NPZ file
        stem0 = Path(path_img0_for_pairfile).stem
        stem1 = Path(path_img1_for_pairfile).stem
        npz_matches_path = os.path.join(DEFAULT_SUPERGLUE_OUTPUT_DIR, f"{stem0}_{stem1}_matches.npz")

        # Display images before running SuperGlue
        print(f"Displaying images for matching: {Path(target_image_full_path).name} and {Path(search_area_image_path).name}")
        display_images_for_matching(target_image_full_path, search_area_image_path)

        # Run SuperGlue
        sg_cmd = [
            "python", DEFAULT_SUPERGLUE_SCRIPT_PATH,
            "--input_pairs", pair_file_path,
            "--input_dir", ".", # Project root
            "--output_dir", DEFAULT_SUPERGLUE_OUTPUT_DIR,
            "--superglue", "outdoor",
            "--viz", # Enable visualization
            "--resize", "-1"
        ]
        success_sg, _ = run_command(sg_cmd)

        est_lat, est_lon = None, None
        if success_sg and os.path.exists(npz_matches_path):
            # Run estimation
            est_cmd = [
                "python", DEFAULT_ESTIMATOR_SCRIPT_PATH,
                npz_matches_path,
                args.search_area_geojson,
                str(args.search_area_img_width), str(args.search_area_img_height),
                str(DEFAULT_TARGET_IMG_WIDTH), str(DEFAULT_TARGET_IMG_HEIGHT)
                # Not passing truth_lat/lon as we just want the estimation
            ]
            success_est, out_est = run_command(est_cmd)
            if success_est:
                est_lat, est_lon = parse_estimation_output_for_coords(out_est)
        else:
            print(f"SuperGlue match failed or NPZ not found for {image_id_filename}.")
        
        all_estimations.append({"image_id": image_id_filename, "latitude": est_lat, "longitude": est_lon})

    # Write all estimations to CSV
    if all_estimations:
        fieldnames = ["image_id", "latitude", "longitude"]
        with open(args.output_estimations_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row_data in all_estimations:
                # Ensure lat/lon are formatted nicely or handle None
                row_data_formatted = {
                    "image_id": row_data["image_id"],
                    "latitude": f"{row_data['latitude']:}" if row_data['latitude'] is not None else "",
                    "longitude": f"{row_data['longitude']:}" if row_data['longitude'] is not None else ""
                }
                writer.writerow(row_data_formatted)
        print(f"\nBatch direct localization complete. Estimations saved to {args.output_estimations_csv}")
    else:
        print("No estimations to save.")

if __name__ == '__main__':
    main()
