import json
from src.utility.bounding_box  import BoundingBox, Projection
from shapely.geometry import shape
from pyproj import Transformer



def geojson_to_3035_bboxes(filepath: str) -> list[BoundingBox]:
    # Load GeoJSON file
    with open(filepath, 'r') as f:
        data = json.load(f)

    transformer = Transformer.from_crs(Projection.EPSG_4326._value_, Projection.EPSG_3035._value_, always_xy=True)

    bboxes_3035: list[BoundingBox] = []

    for feature in data['features']:
        geom = shape(feature['geometry'])

        # Get bounding box in EPSG:4326
        min_lon, min_lat, max_lon, max_lat = geom.bounds

        # Transform to EPSG:3035
        min_lon_3035, min_lat_3035 = transformer.transform(min_lon, min_lat)
        max_lon_3035, max_lat_3035 = transformer.transform(max_lon, max_lat)

        bbox = BoundingBox(
            min_lat=min_lat_3035,
            max_lat=max_lat_3035,
            min_lon=min_lon_3035,
            max_lon=max_lon_3035,
            projection=Projection.EPSG_3035
        )
        bboxes_3035.append(bbox)

    return bboxes_3035

def convert_4326_bbox_to_3035_bbox(min_lon_4326: float, min_lat_4326: float, max_lon_4326: float, max_lat_4326: float) -> BoundingBox:
    """
    Converts an EPSG:4326 bounding box to an EPSG:3035 BoundingBox object.
    """
    transformer = Transformer.from_crs(Projection.EPSG_4326.value, Projection.EPSG_3035.value, always_xy=True)

    # Transform corner points
    min_lon_3035, min_lat_3035 = transformer.transform(min_lon_4326, min_lat_4326)
    max_lon_3035, max_lat_3035 = transformer.transform(max_lon_4326, max_lat_4326)

    # Ensure min <= max after transformation, as projection can sometimes flip coordinates
    # depending on the extent and shape of the area.
    # For typical rectangular areas aligned with lat/lon, this should hold,
    # but it's safer to check.
    actual_min_lon_3035 = min(min_lon_3035, max_lon_3035)
    actual_max_lon_3035 = max(min_lon_3035, max_lon_3035)
    actual_min_lat_3035 = min(min_lat_3035, max_lat_3035)
    actual_max_lat_3035 = max(min_lat_3035, max_lat_3035)


    return BoundingBox(
        min_lat=actual_min_lat_3035,
        max_lat=actual_max_lat_3035,
        min_lon=actual_min_lon_3035,
        max_lon=actual_max_lon_3035,
        projection=Projection.EPSG_3035
    )

if __name__ == "__main__":
    # Test geojson_to_3035_bboxes
    path_to_geojson = "data/example/images/rickmansworth_example_search_area.json"  # Replace with your actual path
    bboxes_from_geojson = geojson_to_3035_bboxes(path_to_geojson)
    for i, bbox_gj in enumerate(bboxes_from_geojson):
        print(f"From GeoJSON - Feature {i+1} BBOX (EPSG:3035): {bbox_gj}")
        print(f"Query string: {bbox_gj.to_query_string()}")

    # Test convert_4326_bbox_to_3035_bbox
    # Example: Eiffel Tower approximate area
    eiffel_min_lon_4326, eiffel_min_lat_4326 = 2.292, 48.856 
    eiffel_max_lon_4326, eiffel_max_lat_4326 = 2.297, 48.860

    print(f"\nTesting direct bbox conversion for Eiffel Tower area:")
    print(f"Input EPSG:4326 bbox: min_lon={eiffel_min_lon_4326}, min_lat={eiffel_min_lat_4326}, max_lon={eiffel_max_lon_4326}, max_lat={eiffel_max_lat_4326}")
    
    converted_bbox = convert_4326_bbox_to_3035_bbox(
        eiffel_min_lon_4326, eiffel_min_lat_4326,
        eiffel_max_lon_4326, eiffel_max_lat_4326
    )
    print(f"Converted BBOX (EPSG:3035): {converted_bbox}")
    print(f"Query string: {converted_bbox.to_query_string()}")
