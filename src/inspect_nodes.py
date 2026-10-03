import zarr
import math
import matplotlib.pyplot as plt

base = "data/train/44b6_0113de3b.geff/nodes"

# Load node IDs
ids = zarr.open(
    f"{base}/ids",
    mode="r"
)[:]

# Load each property
t = zarr.open(
    f"{base}/props/t/values",
    mode="r"
)[:]

z = zarr.open(
    f"{base}/props/z/values",
    mode="r"
)[:]

y = zarr.open(
    f"{base}/props/y/values",
    mode="r"
)[:]

x = zarr.open(
    f"{base}/props/x/values",
    mode="r"
)[:]

# Load tracking edges
edges = zarr.open(
    "data/train/44b6_0113de3b.geff/edges/ids",
    mode="r"
)[:]



node_index = {}



for i in range(len(ids)):
    node_index[ids[i]] = i

# Print each node
for i in range(len(ids)):
    print(
        f"Node {ids[i]}: "
        f"t={t[i]}, "
        f"z={z[i]}, "
        f"y={y[i]}, "
        f"x={x[i]}"
    )


# Physical voxel scales in micrometers
Z_SCALE = 1.625
Y_SCALE = 0.40625
X_SCALE = 0.40625

print("\nTracking edges:")

distances = []

for source_id, target_id in edges:

    source_i = node_index[source_id]
    target_i = node_index[target_id]

    # Difference in voxel coordinates
    dz = z[target_i] - z[source_i]
    dy = y[target_i] - y[source_i]
    dx = x[target_i] - x[source_i]

    # Convert voxel movement to physical movement
    dz_um = dz * Z_SCALE
    dy_um = dy * Y_SCALE
    dx_um = dx * X_SCALE

    # 3D Euclidean distance
    distance = math.sqrt(
        dz_um**2 +
        dy_um**2 +
        dx_um**2
    )

    print(
        f"t={t[source_i]} -> t={t[target_i]} | "
        f"{source_id} -> {target_id} | "
        f"distance={distance:.2f} um"
    )
    distances.append(distance)

# Print summary statistics for distances
if distances:
    print("\nDistance summary:")
    print(f"Number of edges: {len(distances)}")
    print(f"Min distance: {min(distances):.2f} um")
    print(f"Max distance: {max(distances):.2f} um")
    print(f"Average distance: {sum(distances)/len(distances):.2f} um")

    # Plot histogram of distances
    plt.hist(distances, bins=20, edgecolor='black')
    plt.title("Distribution of Tracking Edge Distances")
    plt.xlabel("Distance (um)")
    plt.ylabel("Frequency")
    plt.show()