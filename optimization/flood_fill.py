import numpy as np
import cv2
import time

# Parameters
sz = 2048
image_size = (sz, sz)
disk_center = (sz//2, sz//2)
disk_radius = 20
repeats = 100
 
# Create binary image with a white disk
image = np.zeros(image_size, dtype=np.float32)
cv2.circle(image, disk_center, disk_radius, 1, -1)
         
# --- Benchmark cv2.connectedComponentsWithStats ---
mask = cv2.compare(image, 0.5, cv2.CMP_GT)
start_cc = time.perf_counter()
for _ in range(repeats):
    st = cv2.connectedComponentsWithStats(
    mask, 
    connectivity=8,
    ltype = cv2.CV_16U,
)
end_cc = time.perf_counter()
avg_time_cc = (end_cc - start_cc) / repeats

# --- Benchmark cv2.floodFill ---
mask = np.zeros((sz+2,sz+2), np.uint8)
start_ff = time.perf_counter()
for _ in range(repeats):
    #mask[:] = 0
    cv2.floodFill(
        image, 
        mask, 
        seedPoint=disk_center, 
        newVal = 255, 
        loDiff = 0.1,
        upDiff = 0.1, 
        flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    )
end_ff = time.perf_counter()
avg_time_ff = (end_ff - start_ff) / repeats

# --- Results ---
print(f"Average time over {repeats} runs:")
print(f"  cv2.connectedComponentsWithStats: {avg_time_cc * 1e3:.3f} ms")
print(f"  cv2.floodFill:                    {avg_time_ff * 1e3:.3f} ms")
print(f"Speedup: {avg_time_cc/avg_time_ff:.2f} X")