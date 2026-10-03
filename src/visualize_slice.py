import zarr
import matplotlib.pyplot as plt

# Open the Zarr dataset in read-only mode
root = zarr.open(
    "data/train/44b6_0113de3b.zarr",
    mode="r"
)

# Get the microscopy image array
image = root["0"]

# Select timepoint 0
volume = image[0]

z = 63
# Select the middle Z slice
slice_2d = volume[z]

# maximum intensity projection along the Z axis
#mip = volume.max(axis=0)


# Display the 2D microscopy image
plt.imshow(slice_2d, cmap="gray")
plt.title(f"t = 0, Z = {z}")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Intensity")
plt.show()