import json
scene = "office_0"

with open(f"/mnt/projects/FeatureGSLAM/Replica_v2/vmap/{scene}/habitat/info_semantic.json") as f:
    data = json.load(f)

# Extract unique class IDs from the 'objects' list
class_ids = {obj["class_id"] for obj in data["objects"]}

# Sort them for consistency
sorted_ids = sorted(class_ids)

# Print them as a comma-separated string
print(",".join(map(str, sorted_ids)))
