import argparse
import os
import sys
import json
from pathlib import Path
import math

from PIL import Image, __version__ as PILLOW_VERSION
import matplotlib.pyplot as plt # For optional display
from staticmap import StaticMap
from shapely.geometry import shape, box # For GeoJSON and creating tile bounds

# Monkey-patch for Pillow 10+ if ANTIALIAS is missing, to make staticmap compatible
# staticmap library (as of 0.5.0) uses Image.ANTIALIAS internally.
# Pillow 10.0.0 removed Image.ANTIALIAS, recommending Image.Resampling.LANCZOS or similar.
if PILLOW_VERSION.startswith("10."): # Apply only for Pillow 10.x
    if hasattr(Image, 'Resampling') and not hasattr(Image, 'ANTIALIAS'):
        try:
            Image.ANTIALIAS = Image.Resampling.LANCZOS
            print("DEBUG: Applied monkey-patch for PIL.Image.ANTIALIAS using LANCZOS.") # DEBUG
        except AttributeError:
            # Fallback if somehow Resampling enum itself is different (should not happen for Pillow 10)
            print("DEBUG: Could not apply ANTIALIAS monkey-patch using Resampling.LANCZOS.") # DEBUG
    elif not hasattr(Image, 'Resampling'):
         print("DEBUG: PIL.Image.Resampling not found, cannot apply ANTIALIAS monkey-patch.") # DEBUG


# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

ESRI_WORLD_IMAGERY_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
# For zoom level 19, a common tile size for detailed views is 640x640
DEFAULT_TILE_WIDTH_PX = 640
DEFAULT_TILE_HEIGHT_PX = 640
DEFAULT_ZOOM_LEVEL = 19 # "20-storey high" view, approximately

# Helper function to convert lat/lon/zoom to tile numbers (for understanding, not directly used by staticmap with center/zoom)
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

# Helper function to convert tile numbers to lat/lon bounds of the tile's NW corner
def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def get_tile_geographic_bounds(center_lon, center_lat, zoom, tile_width_px, tile_height_px):
    """
    Estimates the geographic bounding box (EPSG:4326) of a tile rendered by staticmap
    given its center, zoom, and pixel dimensions.
    This is an approximation as staticmap handles the precise rendering.
    The core idea is to find how many degrees one pixel covers at this zoom and latitude.
    """
    # At zoom level 0, the world (360 degrees) is 256 pixels wide.
    # At zoom level z, the world is 256 * (2^z) pixels wide.
    world_pixels_dim_at_zoom = 256 * (2**zoom)
    
    # Degrees per pixel for longitude (constant)
    deg_per_pixel_lon = 360.0 / world_pixels_dim_at_zoom
    
    # Degrees per pixel for latitude (varies with latitude due to Mercator projection)
    # Formula: deg_per_pixel_lat = (360.0 / world_pixels_dim_at_zoom) * cos(center_lat_radians)
    # However, staticmap likely uses a simpler interpretation for its 'zoom' parameter,
    # often aligning with web map tile conventions where zoom levels imply a fixed scale at the equator.
    # For simplicity and consistency with how staticmap might work, we'll use the equatorial deg/px for latitude as well,
    # acknowledging this is an approximation for areas away from the equator.
    deg_per_pixel_lat_approx = 360.0 / world_pixels_dim_at_zoom # Simplified

    half_width_deg = (tile_width_px / 2.0) * deg_per_pixel_lon
    half_height_deg = (tile_height_px / 2.0) * deg_per_pixel_lat_approx # Using simplified lat scaling

    min_lon = center_lon - half_width_deg
    max_lon = center_lon + half_width_deg
    min_lat = center_lat - half_height_deg
    max_lat = center_lat + half_height_deg
    
    return min_lon, min_lat, max_lon, max_lat

def fetch_single_tile(center_lon: float, center_lat: float, zoom: int,
                      tile_pixels_w: int, tile_pixels_h: int,
                      output_image_path: str, display_image: bool = False) -> bool:
    """
    Fetches a single satellite tile using staticmap given center, zoom, and dimensions.
    """
    # print(f"  Fetching tile at center: ({center_lon:.6f}, {center_lat:.6f}), Zoom: {zoom}, Size: {tile_pixels_w}x{tile_pixels_h}") # DEBUG
    try:
        static_map_obj = StaticMap(
            width=tile_pixels_w,
            height=tile_pixels_h,
            url_template=ESRI_WORLD_IMAGERY_URL
        )
        image = static_map_obj.render(center=(center_lon, center_lat), zoom=zoom)
        
        output_dir = os.path.dirname(output_image_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        image.save(output_image_path)
        # print(f"    Tile successfully saved to {output_image_path}") # DEBUG

        if display_image:
            plt.imshow(image)
            plt.title(f"Tile: Center ({center_lon:.4f}, {center_lat:.4f}), Zoom {zoom}")
            plt.axis('off')
            plt.show()
        return True
    except Exception as e:
        print(f"    Error fetching/saving tile for center ({center_lon:.6f}, {center_lat:.6f}), zoom {zoom}: {e}") # DEBUG
        return False

def generate_tiles_for_geojson(
    geojson_path: str,
    output_base_dir: str,
    tile_zoom: int = DEFAULT_ZOOM_LEVEL,
    tile_pixels_w: int = DEFAULT_TILE_WIDTH_PX,
    tile_pixels_h: int = DEFAULT_TILE_HEIGHT_PX,
    overlap_percentage: float = 0.1 # 10% overlap
):
    """
    Generates and fetches satellite tiles covering the area of a GeoJSON.
    Tiles are fetched with a specified zoom level and pixel dimensions.
    """
    # print(f"Starting tile generation for GeoJSON: {geojson_path}") # DEBUG
    # print(f"Output directory: {output_base_dir}") # DEBUG
    # print(f"Tile settings: Zoom={tile_zoom}, Size={tile_pixels_w}x{tile_pixels_h}, Overlap={overlap_percentage*100}%") # DEBUG

    os.makedirs(output_base_dir, exist_ok=True)
    tiles_metadata = []

    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        geom = shape(geojson_data['features'][0]['geometry'])
        min_lon_search, min_lat_search, max_lon_search, max_lat_search = geom.bounds
        # print(f"Search Area Bounds (EPSG:4326): min_lon={min_lon_search:.5f}, min_lat={min_lat_search:.5f}, max_lon={max_lon_search:.5f}, max_lat={max_lat_search:.5f}") # DEBUG

        # Estimate geographic span of one tile to determine step size
        # Using the center of the search area for latitude correction estimation
        avg_search_lat = (min_lat_search + max_lat_search) / 2.0
        
        world_pixels_dim_at_zoom = 256 * (2**tile_zoom)
        deg_per_pixel_lon = 360.0 / world_pixels_dim_at_zoom
        deg_per_pixel_lat_approx = deg_per_pixel_lon * math.cos(math.radians(avg_search_lat))
        if deg_per_pixel_lat_approx == 0: # Avoid division by zero if at pole or error
            deg_per_pixel_lat_approx = deg_per_pixel_lon


        tile_geo_width_approx = tile_pixels_w * deg_per_pixel_lon
        tile_geo_height_approx = tile_pixels_h * deg_per_pixel_lat_approx

        # print(f"  Approx. geographic span of one tile: Width={tile_geo_width_approx:.6f} deg, Height={tile_geo_height_approx:.6f} deg") # DEBUG

        step_lon = tile_geo_width_approx * (1 - overlap_percentage)
        step_lat = tile_geo_height_approx * (1 - overlap_percentage)

        if step_lon <= 0 or step_lat <= 0:
            # print("Error: Tile step size is zero or negative. Check tile dimensions, zoom, and overlap.") # DEBUG
            return {"error": "Tile step size is zero or negative."}


        tile_idx = 0
        # Start from the center of the potential top-leftmost tile that covers the search area's top edge
        current_lat = max_lat_search - (tile_geo_height_approx / 2.0) + (tile_geo_height_approx * overlap_percentage / 2.0)


        while current_lat >= min_lat_search + (tile_geo_height_approx / 2.0) - (tile_geo_height_approx * overlap_percentage / 2.0) :
            # Start from the center of the potential leftmost tile that covers the search area's left edge
            current_lon = min_lon_search + (tile_geo_width_approx / 2.0) - (tile_geo_width_approx * overlap_percentage / 2.0)
            
            row_has_tiles = False
            while current_lon <= max_lon_search - (tile_geo_width_approx / 2.0) + (tile_geo_width_approx * overlap_percentage / 2.0):
                tile_filename = f"tile_{tile_idx:04d}_zoom{tile_zoom}.png" # Simplified name
                tile_output_path = os.path.join(output_base_dir, tile_filename)

                success = fetch_single_tile(
                    center_lon=current_lon,
                    center_lat=current_lat,
                    zoom=tile_zoom,
                    tile_pixels_w=tile_pixels_w,
                    tile_pixels_h=tile_pixels_h,
                    output_image_path=tile_output_path
                )

                if success:
                    row_has_tiles = True
                    tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat = get_tile_geographic_bounds(
                        current_lon, current_lat, tile_zoom, tile_pixels_w, tile_pixels_h
                    )
                    tile_bbox_geojson_coords = [[
                        [tile_min_lon, tile_min_lat], [tile_max_lon, tile_min_lat],
                        [tile_max_lon, tile_max_lat], [tile_min_lon, tile_max_lat],
                        [tile_min_lon, tile_min_lat]
                    ]]
                    
                    tiles_metadata.append({
                        "tile_id": tile_idx,
                        "filename": tile_filename,
                        "absolute_path": os.path.abspath(tile_output_path),
                        "center_lon": current_lon,
                        "center_lat": current_lat,
                        "zoom": tile_zoom,
                        "width_px": tile_pixels_w,
                        "height_px": tile_pixels_h,
                        "estimated_bounds_epsg4326": [tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat],
                        "geojson_feature": { # Storing as a valid GeoJSON feature for each tile
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": tile_bbox_geojson_coords},
                            "properties": {
                                "id": tile_idx, 
                                "image_file": tile_filename,
                                "center_lon": current_lon,
                                "center_lat": current_lat,
                                "zoom": tile_zoom
                            }
                        }
                    })
                    tile_idx += 1
                
                current_lon += step_lon
            
            # If the entire row was outside the main search_area bounds due to large steps/overlap, break
            if not row_has_tiles and current_lon > min_lon_search + (tile_geo_width_approx / 2.0): # check if we made at least one step
                 pass # Allow one row of tiles completely outside if it's the first/last row due to centering logic

            current_lat -= step_lat
            
        metadata_path = os.path.join(output_base_dir, "_tiles_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({"tiles": tiles_metadata, "source_geojson": geojson_path}, f, indent=4)
        # print(f"Tile generation complete. {len(tiles_metadata)} tiles processed. Metadata saved to {metadata_path}") # DEBUG
        return {"metadata_path": metadata_path, "tile_count": len(tiles_metadata)}

    except FileNotFoundError:
        # print(f"Error: GeoJSON file not found at {geojson_path}") # DEBUG
        return {"error": f"GeoJSON file not found at {geojson_path}"}
    except ImportError:
        # print("Error: 'staticmap' or 'shapely' library is not installed.") # DEBUG
        return {"error": "'staticmap' or 'shapely' library is not installed."}
    except Exception as e:
        # print(f"An unexpected error occurred in generate_tiles_for_geojson: {e}") # DEBUG
        # import traceback # DEBUG
        # traceback.print_exc() # DEBUG
        return {"error": f"An unexpected error occurred: {str(e)}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and fetch satellite tiles for a GeoJSON area using StaticMap API.")
    parser.add_argument("geojson_file", help="Path to the input GeoJSON file defining the search area.")
    parser.add_argument("output_directory", help="Directory to save the fetched tiles and metadata file.")
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM_LEVEL, help=f"Zoom level for tiles (default: {DEFAULT_ZOOM_LEVEL}).")
    parser.add_argument("--tile_width", type=int, default=DEFAULT_TILE_WIDTH_PX, help=f"Width of each tile in pixels (default: {DEFAULT_TILE_WIDTH_PX}).")
    parser.add_argument("--tile_height", type=int, default=DEFAULT_TILE_HEIGHT_PX, help=f"Height of each tile in pixels (default: {DEFAULT_TILE_HEIGHT_PX}).")
    parser.add_argument("--overlap", type=float, default=0.1, help="Overlap percentage between tiles (0.0 to 0.9, default: 0.1 for 10%).")
    
    args = parser.parse_args()

    if not (0.0 <= args.overlap < 1.0):
        # print("Error: Overlap must be between 0.0 (inclusive) and 1.0 (exclusive).") # DEBUG
        sys.exit(1)

    result = generate_tiles_for_geojson(
        geojson_path=args.geojson_file,
        output_base_dir=args.output_directory,
        tile_zoom=args.zoom,
        tile_pixels_w=args.tile_width,
        tile_pixels_h=args.tile_height,
        overlap_percentage=args.overlap
    )
    
    if "error" in result:
        print(f"Error during tile generation: {result['error']}")
        sys.exit(1)
    else:
        print(f"Successfully generated {result['tile_count']} tiles. Metadata at: {result['metadata_path']}")
