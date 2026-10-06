import zarr
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


image_path = "data/train/44b6_0113de3b.zarr"

root = zarr.open(image_path, mode="r")
image = root["0"]


def detect_cells(volume):
    threshold = np.percentile(volume, 99)

    mask = volume > threshold

    labels, num_objects = ndimage.label(mask)

    component_sizes = np.bincount(labels.ravel())[1:]

    component_labels = range(1, num_objects + 1)

    centers = ndimage.center_of_mass(
        mask,
        labels,
        component_labels
    )

    filtered_centers = []

    for i, center in enumerate(centers):
        if component_sizes[i] > 10:
            filtered_centers.append(center)

    return filtered_centers



def physical_distance(cell1, cell2):
    z1, y1, x1 = cell1
    z2, y2, x2 = cell2

    dz = (z2 - z1) * 1.625
    dy = (y2 - y1) * 0.40625
    dx = (x2 - x1) * 0.40625

    distance = np.sqrt(
        dz**2 +
        dy**2 +
        dx**2
    )

    return distance

def find_closest_cell(source_cell, target_cells, max_distance=8.0):
    closest_cell = None
    closest_distance = float("inf")

    for target_cell in target_cells:
        distance = physical_distance(source_cell, target_cell)

        if distance < closest_distance:
            closest_distance = distance
            closest_cell = target_cell

    # Only accept the match if it is close enough
    if closest_distance <= max_distance:
        return closest_cell, closest_distance

    return None, closest_distance

def match_cells(source_cells, target_cells, max_distance=8.0):

    distance_matrix = np.zeros(
        (len(source_cells), len(target_cells))
    )

    for i, source_cell in enumerate(source_cells):
        for j, target_cell in enumerate(target_cells):
            distance_matrix[i, j] = physical_distance(
                source_cell,
                target_cell
            )

    row_indices, col_indices = linear_sum_assignment(
        distance_matrix
    )

    matches = []

    for row, col in zip(row_indices, col_indices):
        distance = distance_matrix[row, col]

        if distance <= max_distance:
            matches.append(
                (
                    source_cells[row],
                    target_cells[col],
                    distance
                )
            )

    return matches



# Store detections for each timepoint
detections = {}

for t in range(3):
    volume = image[t]

    centers = detect_cells(volume)

    detections[t] = centers

    print(f"t={t}: {len(centers)} candidates")


ground_truth_t0 = (63, 222, 249)

start_cell, start_error = find_closest_cell(
    ground_truth_t0,
    detections[0]
)

next_cell, movement = find_closest_cell(
    start_cell,
    detections[1]
)

third_cell, movement_2 = find_closest_cell(
    next_cell,
    detections[2]
)

matches_0_to_1 = []
used_targets = set()

for source_cell in detections[0]:

    available_targets = [
        cell for cell in detections[1]
        if cell not in used_targets
    ]

    matched_cell, distance = find_closest_cell(
        source_cell,
        available_targets
    )

    if matched_cell is not None:
        matches_0_to_1.append(
            (source_cell, matched_cell, distance)
        )

        used_targets.add(matched_cell)


cells_t0 = detections[0]
cells_t1 = detections[1]

distance_matrix = np.zeros(
    (len(cells_t0), len(cells_t1))
)

for i, cell_t0 in enumerate(cells_t0):
    for j, cell_t1 in enumerate(cells_t1):
        distance_matrix[i, j] = physical_distance(
            cell_t0,
            cell_t1
        )

print("\nDistance matrix shape:", distance_matrix.shape)

row_indices, col_indices = linear_sum_assignment(distance_matrix)

print("Assignments found:", len(row_indices))

global_matches = []

for row, col in zip(row_indices, col_indices):
    distance = distance_matrix[row, col]

    if distance <= 8.0:
        source_cell = cells_t0[row]
        target_cell = cells_t1[col]

        global_matches.append(
            (source_cell, target_cell, distance)
        )

print("\nGlobal assignment results:")
print("Starting cells:", len(cells_t0))
print("Successful matches:", len(global_matches))
print("Unmatched:", len(cells_t0) - len(global_matches))

ground_truth_t0 = (63, 222, 249)
ground_truth_t1 = (63, 227, 251)

# Find the detection nearest to the known t=0 cell
start_detection, start_error = find_closest_cell(
    ground_truth_t0,
    detections[0]
)

for source_cell, target_cell, distance in global_matches:
    if source_cell == start_detection:

        error_t1 = physical_distance(
            target_cell,
            ground_truth_t1
        )

        print("\nKnown-cell global tracking check:")
        print("t=0 detection:", source_cell)
        print("Matched t=1 detection:", target_cell)
        print("Tracking movement:", distance, "micrometers")
        print("Distance from t=1 ground truth:", error_t1, "micrometers")
        break

print("\nt=0 -> t=1 tracking:")
print("Starting cells:", len(detections[0]))
print("Successful matches:", len(matches_0_to_1))
print("Unmatched:", len(detections[0]) - len(matches_0_to_1))

matched_targets = []

for source_cell, matched_cell, distance in matches_0_to_1:
    matched_targets.append(matched_cell)

unique_targets = set(matched_targets)

print("\nDuplicate match check:")
print("Total matches:", len(matched_targets))
print("Unique target cells:", len(unique_targets))
print("Duplicate assignments:", len(matched_targets) - len(unique_targets))

print("\nTracking test:")
print("t=0 detection:", start_cell)
print("t=1 matched detection:", next_cell)
print("Movement:", movement, "micrometers")
print("t=2 matched detection:", third_cell)
print("Movement t=1 -> t=2:", movement_2, "micrometers")


matches_0_to_1 = match_cells(
    detections[0],
    detections[1]
)

matches_1_to_2 = match_cells(
    detections[1],
    detections[2]
)

print("\nTracking results:")
print("t=0 -> t=1:", len(matches_0_to_1), "matches")
print("t=1 -> t=2:", len(matches_1_to_2), "matches")

ground_truth = {
    0: (63, 222, 249),
    1: (63, 227, 251),
    2: (63, 230, 253)
}

# Find the detected cell nearest to the known t=0 ground truth
current_cell, error_t0 = find_closest_cell(
    ground_truth[0],
    detections[0]
)

print("\nThree-frame tracking check:")
print("t=0:", current_cell, "GT error:", error_t0)

# Follow t=0 -> t=1
next_cell = None

for source, target, distance in matches_0_to_1:
    if source == current_cell:
        next_cell = target
        break

if next_cell is not None:
    error_t1 = physical_distance(
        next_cell,
        ground_truth[1]
    )

    print("t=1:", next_cell, "GT error:", error_t1)

    current_cell = next_cell

    # Follow t=1 -> t=2
    next_cell = None

    for source, target, distance in matches_1_to_2:
        if source == current_cell:
            next_cell = target
            break

    if next_cell is not None:
        error_t2 = physical_distance(
            next_cell,
            ground_truth[2]
        )

        print("t=2:", next_cell, "GT error:", error_t2)
    else:
        print("Track lost between t=1 and t=2")
else:
    print("Track lost between t=0 and t=1")