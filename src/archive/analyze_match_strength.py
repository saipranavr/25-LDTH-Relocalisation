import numpy as np
import argparse

def analyze_matches(npz_path):
    """
    Analyzes the SuperGlue match data from an .npz file and prints statistics.
    """
    try:
        data = np.load(npz_path)
    except FileNotFoundError:
        print(f"Error: File not found at {npz_path}")
        return
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return

    keypoints0 = data.get('keypoints0')
    keypoints1 = data.get('keypoints1')
    matches = data.get('matches') # This corresponds to matches0 in SuperGlue's direct output
    match_confidence = data.get('match_confidence') # This corresponds to matching_scores0

    if keypoints0 is None or matches is None or match_confidence is None:
        print("Error: The .npz file does not contain the expected arrays ('keypoints0', 'matches', 'match_confidence').")
        print(f"Available keys: {list(data.keys())}")
        return

    num_keypoints0 = len(keypoints0)
    num_keypoints1 = len(keypoints1) if keypoints1 is not None else "N/A (not always saved in this format)"


    # Filter out non-matches (where match index is -1)
    actual_match_indices = matches > -1
    num_actual_matches = np.sum(actual_match_indices)

    print(f"Analysis for: {npz_path}")
    print("-" * 30)
    print(f"Total keypoints in Image 0: {num_keypoints0}")
    print(f"Total keypoints in Image 1: {num_keypoints1}")
    print(f"Number of established matches: {num_actual_matches}")

    if num_keypoints0 > 0:
        match_ratio = num_actual_matches / num_keypoints0
        print(f"Percentage of Image 0 keypoints matched: {match_ratio:.2%}")
    else:
        print("Percentage of Image 0 keypoints matched: N/A (no keypoints in Image 0)")

    if num_actual_matches > 0:
        # Get confidence scores for only the actual matches
        confidences_of_actual_matches = match_confidence[actual_match_indices]
        avg_confidence = np.mean(confidences_of_actual_matches)
        min_confidence = np.min(confidences_of_actual_matches)
        max_confidence = np.max(confidences_of_actual_matches)
        
        print(f"Average confidence of established matches: {avg_confidence:.4f}")
        print(f"Min confidence of established matches: {min_confidence:.4f}")
        print(f"Max confidence of established matches: {max_confidence:.4f}")
        
        # Example: Count matches above a certain threshold (e.g., 0.7)
        high_conf_threshold = 0.7
        num_high_confidence_matches = np.sum(confidences_of_actual_matches > high_conf_threshold)
        print(f"Number of matches with confidence > {high_conf_threshold}: {num_high_confidence_matches} ({num_high_confidence_matches/num_actual_matches:.2%})")
    else:
        print("No matches established to calculate confidence statistics.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze SuperGlue match strength from .npz file.')
    parser.add_argument('npz_file_path', type=str, help='Path to the .npz file from SuperGlue.')
    args = parser.parse_args()
    
    analyze_matches(args.npz_file_path)
