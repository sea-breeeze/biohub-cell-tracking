import zarr
import numpy as np
from scipy import ndimage

image_path = "data/train/44b6_0113de3b.zarr"

root = zarr.open(image_path, mode="r")
image = root["0"]

# Load timepoint 0
volume = image[0]

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