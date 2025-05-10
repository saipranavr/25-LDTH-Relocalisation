import json

def create_geojson_search_area(center_lat: float, center_lon: float, delta_lat: float, delta_lon: float, output_filepath: str):
    """
    Creates a GeoJSON file defining a rectangular search area polygon.

    Args:
        center_lat: Center latitude of the search area.
        center_lon: Center longitude of the search area.
        delta_lat: Half-height of the rectangle (latitude difference from center).
        delta_lon: Half-width of the rectangle (longitude difference from center).
        output_filepath: Path to save the generated GeoJSON file.
    """

    min_lon = center_lon - delta_lon
    max_lon = center_lon + delta_lon
    min_lat = center_lat - delta_lat
    max_lat = center_lat + delta_lat

    # Define polygon coordinates: [longitude, latitude]
    # Order: SW, NW, NE, SE, SW (to close)
    coordinates = [
        [min_lon, min_lat],  # Bottom-left
        [min_lon, max_lat],  # Top-left
        [max_lon, max_lat],  # Top-right
        [max_lon, min_lat],  # Bottom-right
        [min_lon, min_lat]   # Closing point
    ]

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates]  # GeoJSON polygons have an outer list for the exterior ring
                }
            }
        ]
    }

    try:
        with open(output_filepath, 'w') as f:
            json.dump(geojson_data, f, indent=2) # indent for readability
        print(f"Successfully created GeoJSON search area at: {output_filepath}")
        print(f"Bounds: Lat ({min_lat:.4f} to {max_lat:.4f}), Lon ({min_lon:.4f} to {max_lon:.4f})")
    except Exception as e:
        print(f"Error writing GeoJSON file: {e}")

if __name__ == "__main__":
    # Eiffel Tower coordinates
    eiffel_tower_lat = 48.8584
    eiffel_tower_lon = 2.2945

    # Define the size of the search area around the Eiffel Tower
    # Approx 0.02 deg lat (~2.2km) x 0.03 deg lon (~2.1km at this latitude)
    latitude_delta_half = 0.01
    longitude_delta_half = 0.015

    output_file = "data/example/images/eiffel_tower_search_area.json"

    create_geojson_search_area(
        center_lat=eiffel_tower_lat,
        center_lon=eiffel_tower_lon,
        delta_lat=latitude_delta_half,
        delta_lon=longitude_delta_half,
        output_filepath=output_file
    )
