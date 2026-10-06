
import zarr
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

image_path = "data/train/44b6_0113de3b.zarr"

root = zarr.open(image_path, mode="r")
image = root["0"]

timepoint = 2

volume = image[timepoint]

ground_truth = {
    0: (63, 222, 249),
    1: (63, 227, 251),
    2: (63, 230, 253)
}

gt_z, gt_y, gt_x = ground_truth[timepoint]

print("Shape:", volume.shape)
print("Min:", volume.min())
print("Max:", volume.max())
print("Mean:", volume.mean())
print("Median:", np.median(volume))

print("90th percentile:", np.percentile(volume, 90))
print("95th percentile:", np.percentile(volume, 95))
print("99th percentile:", np.percentile(volume, 99))
print("99.9th percentile:", np.percentile(volume, 99.9))

threshold = np.percentile(volume, 99)

mask = volume > threshold

print("\nThreshold:", threshold)
print("Bright voxels:", mask.sum())
print("Total voxels:", mask.size)
print("Percent kept:", mask.mean() * 100)

# Group connected bright voxels into separate objects
labels, num_objects = ndimage.label(mask)

component_sizes = np.bincount(labels.ravel())

# Remove label 0 because 0 is the background
component_sizes = component_sizes[1:]

print("\nComponent size summary:")
print("Smallest:", component_sizes.min(), "voxels")
print("Average:", component_sizes.mean(), "voxels")
print("Largest:", component_sizes.max(), "voxels")

print("Components <= 5 voxels:", np.sum(component_sizes <= 5))
print("Components <= 10 voxels:", np.sum(component_sizes <= 10))

print("\nConnected components:", num_objects)

gt_label = labels[gt_z, gt_y, gt_x]

print("\nGround-truth cell check:")
print("Component label:", gt_label)

if gt_label == 0:
    print("Ground-truth location did not survive the threshold.")
else:
    gt_size = component_sizes[gt_label - 1]
    center = ndimage.center_of_mass(mask, labels, gt_label)

    print("Component size:", gt_size)
    print("Component center:", center)

    dz = center[0] - gt_z
    dy = center[1] - gt_y
    dx = center[2] - gt_x

    print("Difference from ground truth:")
    print("dz:", dz)
    print("dy:", dy)
    print("dx:", dx)

# Calculate the center of every detected component
component_labels = range(1, num_objects + 1)

centers = ndimage.center_of_mass(
    mask,
    labels,
    component_labels
)

print("\nNumber of candidate centers:", len(centers))

print("\nFirst 5 candidate centers:")
for i in range(5):
    print(f"Component {i + 1}: {centers[i]}")

    # Filter out very small components
min_size = 10

filtered_centers = []


print("\nFiltering results:")
print("Before filtering:", len(centers))
print("After filtering:", len(filtered_centers))
print("Removed:", len(centers) - len(filtered_centers))

for i, center in enumerate(centers):
    size = component_sizes[i]

    if size > min_size:
        filtered_centers.append(center)

# Create a maximum-intensity projection across Z
mip = volume.max(axis=0)


# # Plot each filtered candidate center
for center in filtered_centers:
    z, y, x = center
    plt.scatter(x, y, s=15, facecolors="none", edgecolors="red")


# Find the candidate center closest to the ground-truth position
closest_center = None
closest_distance = float("inf")

for center in filtered_centers:
    z, y, x = center

    dz = (z - gt_z) * 1.625
    dy = (y - gt_y) * 0.40625
    dx = (x - gt_x) * 0.40625

    distance = np.sqrt(dz**2 + dy**2 + dx**2)

    if distance < closest_distance:
        closest_distance = distance
        closest_center = center

print("\nClosest detected center to ground truth:")
print("Ground truth:", (gt_z, gt_y, gt_x))
print("Detected:", closest_center)
print("Distance:", closest_distance, "micrometers")

plt.imshow(mip, cmap="gray")
plt.title("Detected Cell Candidates")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
