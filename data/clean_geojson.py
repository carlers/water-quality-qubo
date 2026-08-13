import json

# Read the file as text and fix the booleans
with open('LDB_centroids_simplified.geojson', 'r') as f:
    content = f.read()

# Replace Python-style booleans with JSON-style
# Be careful not to alter strings that contain "True" or "False" as text,
# but in this GeoJSON they only appear as standalone values.
content = content.replace('False', 'false').replace('True', 'true')

# Now load the fixed JSON
data = json.loads(content)

# Rename mapping
rename_map = {
    "A1_HexLayer_A1_norm": "a1_river_proximity",
    "A2_HexLayer_A2_norm": "a2_runoff_proximity",
    "B1_HexLayerv2_B1_norm": "b1_fishpen_density",
    "B2_HexLayerv3_B2_norm": "b2_hypoxic_proximity",
    "C1_HexMap_C1_norm": "c1_bathymetric_depressions",
    "C2_HexLayer_C2_norm": "c2_outlet_proximity",
    "D1_HexLayerPolygon \u00e2\u20ac\u201d LDB_clippedHEX_D1_HexLayer_D1_norm": "d1_boatramp_proximity",
    "D2_HexLayerv2_D2_norm": "d2_road_proximity",
    "LDB_existing_station_hexmap_has_point_bool": "has_existing_station"
}

for feature in data['features']:
    props = feature['properties']
    for old, new in rename_map.items():
        if old in props:
            props[new] = props.pop(old)

# Save clean file
with open('LDB_centroids_clean.geojson', 'w') as f:
    json.dump(data, f, indent=2)

print("✅ Done! Cleaned file saved as 'LDB_centroids_clean.geojson'")