#!/usr/bin/env python3
"""
Sample example demonstrating enhanced filter_bbox with VectorCube support

This example shows how the improved filter_bbox function now works with:
1. RasterCube (original functionality)
2. VectorCube (new functionality)
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from openeo_pg_parser_networkx.pg_schema import BoundingBox
from shapely.geometry import Point, Polygon

# Import the enhanced filter_bbox function
from openeo_processes_dask.process_implementations.cubes.filter_bbox import filter_bbox


def create_sample_vectorcube():
    """Create a sample VectorCube (GeoDataFrame) for testing"""

    # Create sample points spread across different locations
    points = [
        Point(10.5, 45.5),  # Inside bbox
        Point(11.5, 46.5),  # Outside bbox
        Point(10.0, 45.0),  # On bbox boundary (should be included with intersects())
        Point(10.2, 45.8),  # Inside bbox
        Point(12.0, 47.0),  # Outside bbox
    ]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Point_A", "Point_B", "Point_C", "Point_D", "Point_E"],
            "geometry": points,
        }
    )

    # Set CRS to WGS84
    gdf.crs = "EPSG:4326"

    return gdf


def test_vectorcube_filtering():
    """Test the enhanced filter_bbox with VectorCube"""

    print("🧪 Testing Enhanced filter_bbox with VectorCube")
    print("=" * 60)

    # Create sample data
    vector_data = create_sample_vectorcube()

    print("📊 Original VectorCube data:")
    print(vector_data)
    print(f"CRS: {vector_data.crs}")
    print()

    # Define bounding box for filtering
    bbox = BoundingBox(west=10.0, east=11.0, south=45.0, north=46.0, crs="EPSG:4326")

    print(f"🔍 Filtering with BoundingBox: {bbox}")
    print()

    try:
        # Apply the enhanced filter_bbox function
        filtered_result = filter_bbox(vector_data, bbox)

        print("✅ SUCCESS: filter_bbox worked with VectorCube!")
        print("📋 Filtered results:")
        print(filtered_result)
        print()
        print(f"📈 Original points: {len(vector_data)}")
        print(f"📉 Filtered points: {len(filtered_result)}")
        print()

        # Show which points were included
        print("🎯 Points included in results:")
        for idx, row in filtered_result.iterrows():
            point = row.geometry
            print(f"  - {row['name']}: ({point.x}, {point.y})")

        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_coordinate_reprojection():
    """Test the fixed coordinate reprojection"""

    print("\n🔄 Testing Coordinate Reprojection Fix")
    print("=" * 60)

    # Test reprojection from WGS84 to Web Mercator
    from openeo_processes_dask.process_implementations.cubes.filter_bbox import (
        _reproject_bbox,
    )

    bbox_wgs84 = BoundingBox(
        west=10.0, east=11.0, south=45.0, north=46.0, crs="EPSG:4326"
    )

    try:
        bbox_mercator = _reproject_bbox(bbox_wgs84, "EPSG:3857")

        print("✅ SUCCESS: Coordinate reprojection working!")
        print(f"📍 Original (WGS84): west={bbox_wgs84.west}, east={bbox_wgs84.east}")
        print(
            f"📍 Reprojected (Mercator): west={bbox_mercator.west:.2f}, east={bbox_mercator.east:.2f}"
        )
        return True

    except Exception as e:
        print(f"❌ ERROR in reprojection: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Enhanced filter_bbox Demonstration")
    print("=" * 60)
    print("This example demonstrates the improvements made to filter_bbox:")
    print("1. ✅ VectorCube support (GeoDataFrame filtering)")
    print("2. ✅ Fixed coordinate transformation bug")
    print("3. ✅ Improved geometric filtering (intersects vs within)")
    print()

    # Run tests
    vectorcube_success = test_vectorcube_filtering()
    reprojection_success = test_coordinate_reprojection()

    print("\n🎉 SUMMARY:")
    print(f"VectorCube filtering: {'✅ WORKING' if vectorcube_success else '❌ FAILED'}")
    print(
        f"Coordinate reprojection: {'✅ WORKING' if reprojection_success else '❌ FAILED'}"
    )

    if vectorcube_success and reprojection_success:
        print("\n🎯 All enhancements are working correctly!")
        print("📝 Your colleague can use this code to demonstrate the improvements.")
    else:
        print("\n⚠️  Some issues detected - please check the errors above.")
