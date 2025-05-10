import csv
import argparse
import os
from math import radians, sin, cos, sqrt, atan2
from typing import List, Tuple, Any
from tabulate import tabulate
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

def load_csv(filepath: str) -> List[dict[str, Any]]:
    with open(filepath, newline='') as csvfile:
        return list(csv.DictReader(csvfile))

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371e3  # meters
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)

    a = sin(d_phi / 2)**2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def compare_guesses_to_truth(guesses_path: str, truth_path: str) -> Tuple[List[List[str]], float, int, int]:
    guesses = load_csv(guesses_path)
    truths = load_csv(truth_path)

    guess_dict = {row['image_id']: row for row in guesses}
    truth_dict = {row['image_id']: row for row in truths}

    results: List[List[str]] = []
    total_error = 0.0
    match_count = 0
    not_supplied_count = 0

    for image_id in truth_dict:
        if image_id in guess_dict:
            try:
                lat1 = float(truth_dict[image_id]['latitude'])
                lon1 = float(truth_dict[image_id]['longitude'])
                lat2 = float(guess_dict[image_id]['latitude'])
                lon2 = float(guess_dict[image_id]['longitude'])
                error = haversine(lat1, lon1, lat2, lon2)

                results.append([
                    image_id,
                    f"{lat1:.6f}, {lon1:.6f}",
                    f"{lat2:.6f}, {lon2:.6f}",
                    f"{error:.2f} m"
                ])
                total_error += error
                match_count += 1
            except ValueError:
                print(f"[WARN] Invalid lat/lon for image_id {image_id}")
        else:
            print(f"[WARN] No guess provided for image_id {image_id}")
            results.append([
                image_id,
                f"{truth_dict[image_id]['latitude']}, {truth_dict[image_id]['longitude']}",
                "N/A",
                "N/A"
            ])
            not_supplied_count += 1

    return results, total_error, match_count, not_supplied_count

def main() -> None:
    import os
    import sys
    # Ensure the src directory is in the Python path to allow imports like 'from src.module import func'
    # This might need adjustment depending on how the script is run.
    # If evaluate_estimations.py is in the root, and src is a subdir, this should work.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == "src" else current_dir
    if project_root not in sys.path:
         sys.path.append(project_root)

    from src.sift_baseline import sift_baseline # Keep for comparison or alternative run
    from src.superpoint_superglue_matching import relocalize_drone
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate localization estimations against a truth file.')
    parser.add_argument('data_directory', help='Path to the data directory (e.g., data/example or data/test)')
    parser.add_argument('--method', type=str, default='superpoint_superglue', choices=['sift', 'superpoint_superglue'],
                        help='Localization method to use.')
    args = parser.parse_args()

    input_estimations_path = os.path.join(args.data_directory, "estimations.csv") # This is the initial guess file
    truth_path = os.path.join(args.data_directory, "truth.csv")
    images_dir = os.path.join(args.data_directory, "images")

    if not os.path.isfile(input_estimations_path):
        print(f"❌ Initial estimations file not found: {input_estimations_path}")
        print("This file should contain initial guesses (image_id, longitude, latitude).")
        return
    if not os.path.isfile(truth_path):
        print(f"❌ Truth file not found: {truth_path}")
        return
    if not os.path.isdir(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return

    initial_guesses = load_csv(input_estimations_path)
    output_estimations_data = []

    print(f"\nProcessing with method: {args.method}")

    for guess_row in initial_guesses:
        image_id = guess_row['image_id']
        try:
            initial_lon = float(guess_row['longitude'])
            initial_lat = float(guess_row['latitude'])
        except ValueError:
            print(f"[WARN] Skipping {image_id} due to invalid initial lon/lat: {guess_row.get('longitude')}, {guess_row.get('latitude')}")
            output_estimations_data.append({'image_id': image_id, 'longitude': 'N/A', 'latitude': 'N/A'})
            continue

        # Construct drone image path, assuming .jpg extension for now
        # This might need to be more robust if extensions vary or are in the image_id
        drone_image_filename = f"{image_id}.jpg" # Or read from a manifest if complex
        drone_image_path = os.path.join(images_dir, drone_image_filename)

        if not os.path.isfile(drone_image_path):
            print(f"[WARN] Drone image not found for {image_id} at {drone_image_path}. Skipping.")
            output_estimations_data.append({'image_id': image_id, 'longitude': 'N/A', 'latitude': 'N/A'})
            continue
        
        print(f"\nProcessing {image_id}...")
        predicted_lon, predicted_lat = None, None

        if args.method == 'sift':
            # SIFT baseline might need a search_area_json_path, adapt if necessary
            # For this example, let's assume sift_baseline can also take initial lon/lat
            # or that its current implementation in evaluate_estimations is what's desired for SIFT.
            # The original sift_baseline call in this script was:
            # search_area_json_path = os.path.join(args.data_directory, "images", f"{image_id}_search_area.json") # Assuming a convention
            # if not os.path.exists(search_area_json_path): # SIFT needs this
            #     print(f"Warning: search area JSON not found for {image_id} for SIFT. Using placeholder logic if any.")
            #     # Fallback or skip for SIFT if JSON is mandatory and not found
            #     # For now, let's assume sift_baseline is self-contained for its example run
            #     # This part needs to align with how sift_baseline is intended to be used in evaluation.
            #     # The original code called sift_baseline with hardcoded "rickmansworth_example" paths.
            #     # To make it generic:
            #     # predicted_lon, predicted_lat = sift_baseline(drone_image_path, search_area_json_path)
            print("SIFT method selected. Using placeholder logic from original script for now.")
            # This is a simplification; sift_baseline would need its specific inputs.
            # The original script hardcoded paths for sift_baseline.
            # If we want to run SIFT for any image_id, sift_baseline needs to be more generic
            # or we need to provide its specific inputs (like search_area_json_path) per image_id.
            if image_id == "rickmansworth_example": # Replicating original SIFT call for this specific ID
                 sift_uav_image_path = os.path.join(images_dir, "rickmansworth_example.jpg")
                 sift_search_area_json_path = os.path.join(images_dir, "rickmansworth_example_search_area.json")
                 if os.path.exists(sift_uav_image_path) and os.path.exists(sift_search_area_json_path):
                    predicted_lon, predicted_lat = sift_baseline(sift_uav_image_path, sift_search_area_json_path)
                 else:
                    print(f"Required files for SIFT baseline on {image_id} not found. Skipping SIFT.")
                    predicted_lon, predicted_lat = initial_lon, initial_lat # Fallback
            else:
                print(f"SIFT baseline not configured for generic image_id '{image_id}'. Using initial guess.")
                predicted_lon, predicted_lat = initial_lon, initial_lat


        elif args.method == 'superpoint_superglue':
            # Construct the path to the search area GeoJSON, assuming a convention like image_id_search_area.json
            # For the specific example, it's 'rickmansworth_example_search_area.json'
            if image_id == "rickmansworth_example":
                search_area_geojson_path = os.path.join(images_dir, "rickmansworth_example_search_area.json")
            else:
                # Fallback or error for other image_ids if no generic way to find their GeoJSON
                print(f"Warning: No specific GeoJSON search area defined for {image_id}. Using a placeholder or skipping.")
                # As a placeholder, we might try to use the initial_lon/lat to define a small search area,
                # but this would revert to the previous limited search.
                # For now, let's make it clear this example primarily supports rickmansworth_example fully.
                # Or, we could try to make relocalize_drone handle a None geojson_path by using initial_est_lon/lat + radius.
                # However, the goal is to use the wide search area.
                # For this iteration, let's assume if not rickmansworth, it might not work well.
                # A better solution would be a manifest file linking image_id to its specific geojson.
                search_area_geojson_path = os.path.join(images_dir, f"{image_id}_search_area.json") # Generic attempt

            if not os.path.isfile(search_area_geojson_path):
                print(f"❌ Search area GeoJSON not found: {search_area_geojson_path} for image_id {image_id}")
                print("SuperPoint+SuperGlue method requires this file to define the search extent.")
                predicted_lon, predicted_lat = -999.0, -999.0 # Indicate failure
            else:
                predicted_lon, predicted_lat = relocalize_drone(
                    drone_image_path=drone_image_path,
                    search_area_geojson_path=search_area_geojson_path
                    # Add other parameters for relocalize_drone if needed (grid_points, etc.)
                )
        
        output_estimations_data.append({
            'image_id': image_id,
            'longitude': f"{predicted_lon:.8f}" if predicted_lon is not None else 'N/A',
            'latitude': f"{predicted_lat:.8f}" if predicted_lat is not None else 'N/A'
        })

    # Write the new estimations to a method-specific CSV file
    output_filename = f"estimations_generated_{args.method}.csv"
    output_estimations_path = os.path.join(args.data_directory, output_filename)
    
    try:
        with open(output_estimations_path, 'w', newline='') as csvfile:
            fieldnames = ['image_id', 'longitude', 'latitude']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in output_estimations_data:
                writer.writerow(row)
        print(f"\nGenerated estimations saved to: {output_estimations_path}")
    except IOError:
        print(f"Error: Could not write to {output_estimations_path}")
        return

    # Compare the newly generated estimations file with the truth
    results, total_error, match_count, not_supplied_count = compare_guesses_to_truth(output_estimations_path, truth_path)

    headers = ['Image ID', 'Truth (lat, lon)', 'Guess (lat, lon)', 'Error']
    print("\n")
    print(tabulate(results, headers=headers, tablefmt='pretty'))
    print(f"\nTotal error: {total_error:.2f} m over {match_count} matches (Method: {args.method})\n")
    if not_supplied_count > 0:
        print(f"{not_supplied_count} estimation(s) were not provided or failed processing\n")

if __name__ == '__main__':
    main()
