import numpy as np
import cv2
import time

# Parameters
sz = 100
image_size = (sz, sz)
disk_center = (sz//2, sz//2)
disk_radius = 20
repeats = 100
 
# Create binary image with a white disk
image = np.zeros(image_size, dtype=np.float32)
cv2.circle(image, disk_center, disk_radius, 1, -1)
         
# --- Benchmark cv2.connectedComponentsWithStats ---
start_cc = time.perf_counter()
for _ in range(repeats):
    mask = cv2.compare(image, 0.5, cv2.CMP_GT)
    n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, 
        connectivity=8,
        ltype = cv2.CV_16U,
    )
    centroid = centroids[1,:]
end_cc = time.perf_counter()
avg_time_cc = (end_cc - start_cc) / repeats

# --- Benchmark cv2.floodFill ---
start_ff = time.perf_counter()
for _ in range(repeats):
    mask = np.zeros((sz+2,sz+2), np.uint8)
    flat_index = np.argmax(image)
    y, x = np.unravel_index(flat_index, image.shape)
    num_pixels_filled, image, mask, rect = cv2.floodFill(
        image, 
        mask, 
        seedPoint = (x,y), 
        newVal = 255, 
        loDiff = 0.1,
        upDiff = 0.1, 
        flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    )
    centroid = (rect[0]+rect[2]//2, rect[1]+ rect[3]//2)
end_ff = time.perf_counter()
avg_time_ff = (end_ff - start_ff) / repeats

# --- Results ---
print(f"Average time over {repeats} runs:")
print(f"  cv2.connectedComponentsWithStats: {avg_time_cc * 1e3:.3f} ms")
print(f"  cv2.floodFill:                    {avg_time_ff * 1e3:.3f} ms")
print(f"Speedup: {avg_time_cc/avg_time_ff:.2f} X")