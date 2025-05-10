import cv2
import argparse
from pathlib import Path

def downsample_image(input_path: str, output_path: str, factor: int):
    """
    Reads an image, downsamples it by an integer factor, and saves it.
    """
    try:
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Could not read image from {input_path}")
            return
    except Exception as e:
        print(f"Error reading image {input_path}: {e}")
        return

    new_width = img.shape[1] // factor
    new_height = img.shape[0] // factor

    if new_width == 0 or new_height == 0:
        print(f"Error: Downsampling factor {factor} is too large for image size {img.shape[1]}x{img.shape[0]}. Results in zero dimension.")
        return

    downsampled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, downsampled_img)
        print(f"Successfully downsampled '{input_path}' by factor {factor} and saved to '{output_path}' ({new_width}x{new_height})")
    except Exception as e:
        print(f"Error writing image {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample an image by an integer factor.")
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument("output_image", help="Path to save the downsampled image.")
    parser.add_argument("downsample_factor", type=int, help="Integer factor by which to downsample the image (e.g., 2, 4).")
    
    args = parser.parse_args()

    if args.downsample_factor <= 0:
        print("Error: Downsample factor must be a positive integer.")
    else:
        downsample_image(args.input_image, args.output_image, args.downsample_factor)
