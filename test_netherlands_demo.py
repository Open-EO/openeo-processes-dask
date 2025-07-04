#!/usr/bin/env python3
"""
Test script to demonstrate VectorCube filtering with filter_bbox function.
This script shows how the enhanced filter_bbox works with real GeoJSON data,
using the Netherlands polygon for realistic demonstration.
"""

import os

import geopandas as gpd
import pandas as pd
from openeo_pg_parser_networkx.pg_schema import BoundingBox
from shapely.geometry import Point, Polygon

from openeo_processes_dask.process_implementations.cubes.filter_bbox import filter_bbox


def load_netherlands_data():
    """Load the Netherlands GeoJSON data."""
    geojson_path = "Netherlands_polygon.geojson"
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    gdf = gpd.read_file(geojson_path)
    print(f"Loaded Netherlands data: {len(gdf)} features")
    print(f"CRS: {gdf.crs}")
    print(f"Columns: {list(gdf.columns)}")
    print(f"Geometry type: {gdf.geometry.type.iloc[0]}")

    # Get the bounds of the Netherlands
    bounds = gdf.total_bounds
    print(f"Netherlands bounds: {bounds} (minx, miny, maxx, maxy)")

    return gdf


def create_test_points():
    """Create test points inside and outside Netherlands for demonstration."""
    # Create test points - some inside Netherlands, some outside
    points_data = [
        {"id": 1, "name": "Amsterdam", "lat": 52.3676, "lon": 4.9041},  # Inside
        {"id": 2, "name": "Utrecht", "lat": 52.0907, "lon": 5.1214},  # Inside
        {"id": 3, "name": "The_Hague", "lat": 52.0705, "lon": 4.3007},  # Inside
        {"id": 4, "name": "Rotterdam", "lat": 51.9244, "lon": 4.4777},  # Inside
        {"id": 5, "name": "Groningen", "lat": 53.2194, "lon": 6.5665},  # Inside
        {"id": 6, "name": "Maastricht", "lat": 50.8514, "lon": 5.6909},  # Inside
        {"id": 7, "name": "Berlin", "lat": 52.5200, "lon": 13.4050},  # Outside
        {"id": 8, "name": "Paris", "lat": 48.8566, "lon": 2.3522},  # Outside
        {"id": 9, "name": "London", "lat": 51.5074, "lon": -0.1278},  # Outside
        {
            "id": 10,
            "name": "Brussels",
            "lat": 50.8503,
            "lon": 4.3517,
        },  # Outside (close)
    ]

    # Create Point geometries
    geometries = [Point(row["lon"], row["lat"]) for row in points_data]

    # Create GeoDataFrame (this is a VectorCube)
    gdf = gpd.GeoDataFrame(points_data, geometry=geometries, crs="EPSG:4326")

    return gdf


def main():
    """Main function to test VectorCube filtering with real Netherlands data."""
    print("=== VectorCube Filter BBox with Real Netherlands Data ===\n")

    # Load Netherlands polygon
    print("1. Loading Netherlands GeoJSON data...")
    try:
        netherlands_gdf = load_netherlands_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    print()

    # Create test points
    print("2. Creating test points (Dutch cities and international cities)...")
    points_gdf = create_test_points()

    print(f"Test points (CRS: {points_gdf.crs}):")
    for _, row in points_gdf.iterrows():
        print(
            f"   {row['id']:2d}. {row['name']:12s} ({row['lat']:7.4f}, {row['lon']:7.4f})"
        )
    print()

    # Get Netherlands bounds for bbox filtering
    bounds = netherlands_gdf.total_bounds
    # Add small margin to ensure we include boundary points
    margin = 0.1
    bbox_coords = [
        bounds[0] - margin,
        bounds[1] - margin,
        bounds[2] + margin,
        bounds[3] + margin,
    ]

    print(f"3. Netherlands bounding box (with {margin}° margin):")
    print(
        f"   west={bbox_coords[0]:.2f}, south={bbox_coords[1]:.2f}, east={bbox_coords[2]:.2f}, north={bbox_coords[3]:.2f}"
    )
    print()

    # Create BoundingBox object
    bbox = BoundingBox(
        west=bbox_coords[0],
        south=bbox_coords[1],
        east=bbox_coords[2],
        north=bbox_coords[3],
        crs="EPSG:4326",
    )

    # Apply filter_bbox
    print("4. Applying filter_bbox to test points...")
    filtered_gdf = filter_bbox(points_gdf, bbox)

    # Show results
    print(f"Filtered results (CRS: {filtered_gdf.crs}):")
    for _, row in filtered_gdf.iterrows():
        print(
            f"   {row['id']:2d}. {row['name']:12s} ({row['lat']:7.4f}, {row['lon']:7.4f})"
        )
    print()

    print("5. Summary:")
    print(f"   Original points: {len(points_gdf)}")
    print(f"   Points within Netherlands bbox: {len(filtered_gdf)}")
    print(f"   Points filtered out: {len(points_gdf) - len(filtered_gdf)}")

    # Show which points were filtered out
    filtered_ids = set(filtered_gdf["id"])
    excluded_points = points_gdf[~points_gdf["id"].isin(filtered_ids)]
    if len(excluded_points) > 0:
        print(f"   Excluded points: {', '.join(excluded_points['name'])}")
    print()

    # Test coordinate reprojection
    print("6. Testing coordinate reprojection with Dutch national grid (EPSG:28992)...")

    # Convert test points to Dutch national coordinate system (Amersfoort / RD New)
    points_rd = points_gdf.to_crs("EPSG:28992")

    print(f"   Test points converted to Dutch RD coordinates (EPSG:28992)")
    print(f"   Sample RD coordinates for Amsterdam: {points_rd.iloc[0].geometry}")
    print()

    # Apply same geographic bbox to RD data (should auto-reproject)
    print("   Applying same geographic bbox to RD-projected data...")
    filtered_rd_gdf = filter_bbox(points_rd, bbox)

    print(f"   Filtered RD data (CRS: {filtered_rd_gdf.crs}):")
    print(f"   Number of points: {len(filtered_rd_gdf)}")
    print()

    # Verify that both filtering approaches give same results
    original_names = set(filtered_gdf["name"])
    rd_names = set(filtered_rd_gdf["name"])

    print("7. Verification:")
    if original_names == rd_names:
        print("   ✅ SUCCESS: Both filtering approaches returned identical results!")
        print(f"   Points included: {sorted(original_names)}")
    else:
        print("   ❌ ERROR: Different results between CRS approaches")
        print(f"   WGS84 result: {sorted(original_names)}")
        print(f"   RD result: {sorted(rd_names)}")

    print()

    # Test filtering the Netherlands polygon itself
    print("8. Testing filter_bbox on the Netherlands polygon...")

    # Use a smaller bbox that should only capture part of the Netherlands
    partial_bbox_coords = [4.0, 51.0, 6.0, 53.0]  # Western part of Netherlands
    partial_bbox = BoundingBox(
        west=partial_bbox_coords[0],
        south=partial_bbox_coords[1],
        east=partial_bbox_coords[2],
        north=partial_bbox_coords[3],
        crs="EPSG:4326",
    )
    print(f"   Using partial bbox: {partial_bbox_coords}")

    filtered_netherlands_gdf = filter_bbox(netherlands_gdf, partial_bbox)

    print(f"   Original Netherlands features: {len(netherlands_gdf)}")
    print(f"   Features intersecting partial bbox: {len(filtered_netherlands_gdf)}")

    if len(filtered_netherlands_gdf) > 0:
        print("   ✅ Netherlands polygon successfully filtered with partial bbox")
    else:
        print(
            "   ⚠️  No features found - this might indicate an issue with the filtering"
        )

    print()
    print("=== Real Data Demonstration Complete ===")


if __name__ == "__main__":
    main()
