import json
from pyproj import Transformer
import argparse # For new main functionality
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utility.bounding_box import Projection # For Projection enum

def create_geojson_from_epsg4326_bounds(
    min_lon_4326: float, min_lat_4326: float, 
    max_lon_4326: float, max_lat_4326: float, 
    output_filepath: str
    ):
    """Creates a GeoJSON file from EPSG:4326 bounding box coordinates."""
    
    coordinates_4326 = [
        [min_lon_4326, min_lat_4326],
        [min_lon_4326, max_lat_4326],
        [max_lon_4326, max_lat_4326],
        [max_lon_4326, min_lat_4326],
        [min_lon_4326, min_lat_4326]
    ]
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates_4326] 
                }
            }
        ]
    }
    try:
        with open(output_filepath, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        print(f"Successfully created GeoJSON search area at: {output_filepath}")
        print(f"EPSG:4326 Bounds: Lat ({min_lat_4326:.6f} to {max_lat_4326:.6f}), Lon ({min_lon_4326:.6f} to {max_lon_4326:.6f})")
    except Exception as e:
        print(f"Error writing GeoJSON file: {e}")


def main_create_from_center_delta_epsg4326(center_lat, center_lon, delta_lat, delta_lon, output_filepath):
    """Original functionality: creates GeoJSON from center and deltas in EPSG:4326."""
    min_lon_4326 = center_lon - delta_lon
    max_lon_4326 = center_lon + delta_lon
    min_lat_4326 = center_lat - delta_lat
    max_lat_4326 = center_lat + delta_lat
    create_geojson_from_epsg4326_bounds(min_lon_4326, min_lat_4326, max_lon_4326, max_lat_4326, output_filepath)


def main_create_from_epsg3035_bounds(min_lon_3035, min_lat_3035, max_lon_3035, max_lat_3035, output_filepath):
    """New functionality: creates GeoJSON from EPSG:3035 bounds by converting them to EPSG:4326 first."""
    transformer_3035_to_4326 = Transformer.from_crs(Projection.EPSG_3035.value, Projection.EPSG_4326.value, always_xy=True)
    
    # Transform corner points
    # SW
    sw_lon_4326, sw_lat_4326 = transformer_3035_to_4326.transform(min_lon_3035, min_lat_3035)
    # NW
    # nw_lon_4326, nw_lat_4326 = transformer_3035_to_4326.transform(min_lon_3035, max_lat_3035) # Not needed for bbox
    # NE
    ne_lon_4326, ne_lat_4326 = transformer_3035_to_4326.transform(max_lon_3035, max_lat_3035)
    # SE
    # se_lon_4326, se_lat_4326 = transformer_3035_to_4326.transform(max_lon_3035, min_lat_3035) # Not needed for bbox

    # The resulting EPSG:4326 shape might not be perfectly rectangular, 
    # but for GeoJSON bbox, we just need min/max lat/lon.
    # However, for a polygon feature, we should transform all 4 corners.
    # For simplicity here, we'll assume the transformed min/max are sufficient for the polygon.
    # A more robust way would be to transform all 4 corners of the 3035 box and then find the envelope in 4326.
    # Or, construct the polygon from the 4 transformed 3035 corners.

    # For this example, we'll use the transformed SW and NE corners to define the 4326 bbox.
    # This is an approximation if the transformed shape isn't axis-aligned in 4326.
    # A better way is to transform all 4 corners of the 3035 box.
    
    # Let's transform all 4 corners for the polygon
    points_3035 = [
        (min_lon_3035, min_lat_3035), # SW
        (min_lon_3035, max_lat_3035), # NW
        (max_lon_3035, max_lat_3035), # NE
        (max_lon_3035, min_lat_3035)  # SE
    ]
    
    points_4326 = []
    for lon3035, lat3035 in points_3035:
        lon4326, lat4326 = transformer_3035_to_4326.transform(lon3035, lat3035)
        points_4326.append([lon4326, lat4326])
    
    # Close the polygon
    points_4326.append(list(points_4326[0])) # Make a copy to avoid modifying the first point if it's a list

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [points_4326] 
                }
            }
        ]
    }
    try:
        with open(output_filepath, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        print(f"Successfully created GeoJSON search area at: {output_filepath} from EPSG:3035 bounds.")
        # Print the approximate 4326 bounds for info
        lons_4326 = [p[0] for p in points_4326[:-1]]
        lats_4326 = [p[1] for p in points_4326[:-1]]
        print(f"Approx EPSG:4326 Bounds: Lat ({min(lats_4326):.6f} to {max(lats_4326):.6f}), Lon ({min(lons_4326):.6f} to {max(lons_4326):.6f})")

    except Exception as e:
        print(f"Error writing GeoJSON file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GeoJSON search area.")
    parser.add_argument("--mode", choices=["center_delta_4326", "bounds_3035"], default="center_delta_4326", 
                        help="Mode of operation: create from EPSG:4326 center/delta or from EPSG:3035 bounds.")
    
    # Args for center_delta_4326 mode
    parser.add_argument("--center_lat", type=float, help="Center latitude (EPSG:4326).")
    parser.add_argument("--center_lon", type=float, help="Center longitude (EPSG:4326).")
    parser.add_argument("--delta_lat", type=float, help="Latitude delta/half-height (degrees).")
    parser.add_argument("--delta_lon", type=float, help="Longitude delta/half-width (degrees).")

    # Args for bounds_3035 mode
    parser.add_argument("--min_lon_3035", type=float, help="Min longitude (EPSG:3035).")
    parser.add_argument("--min_lat_3035", type=float, help="Min latitude (EPSG:3035).")
    parser.add_argument("--max_lon_3035", type=float, help="Max longitude (EPSG:3035).")
    parser.add_argument("--max_lat_3035", type=float, help="Max latitude (EPSG:3035).")
    
    parser.add_argument("output_filepath", help="Path to save the generated GeoJSON file.")
    
    args = parser.parse_args()

    if args.mode == "center_delta_4326":
        if not all([args.center_lat, args.center_lon, args.delta_lat, args.delta_lon]):
            parser.error("For mode 'center_delta_4326', --center_lat, --center_lon, --delta_lat, and --delta_lon are required.")
        main_create_from_center_delta_epsg4326(args.center_lat, args.center_lon, args.delta_lat, args.delta_lon, args.output_filepath)
    elif args.mode == "bounds_3035":
        if not all([args.min_lon_3035, args.min_lat_3035, args.max_lon_3035, args.max_lat_3035]):
            parser.error("For mode 'bounds_3035', --min_lon_3035, --min_lat_3035, --max_lon_3035, and --max_lat_3035 are required.")
        main_create_from_epsg3035_bounds(args.min_lon_3035, args.min_lat_3035, args.max_lon_3035, args.max_lat_3035, args.output_filepath)
