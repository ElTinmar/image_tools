import numpy as np
import cv2
import time
from .morphology import (
    filter_contours, 
    filter_contours_centroids, 
    filter_connected_comp,
    filter_connected_comp_centroids,
    filter_floodfill,
    filter_floodfill_centroid
)

# Parameters
sizes = 2**np.arange(4,12)

for sz in sizes:

    image_size = (sz, sz)
    ellipse_center = (sz//4, sz//6)
    axes_length = (sz//10, sz//14) 
    ellipse_angle = 30
    repeats = 500
    
    # Create binary image with a white ellipse
    image = np.zeros(image_size, dtype=np.float32)
    image = cv2.ellipse(image, ellipse_center, axes_length, ellipse_angle, 0, 360, 1, -1)
    
    # --- cv2.connectedComponentsWithStats ---
    start_cc = time.perf_counter()
    for _ in range(repeats):
        mask = cv2.compare(image, 0.5, cv2.CMP_GT)
        filter_connected_comp(mask, max_size=200_000)
    end_cc = time.perf_counter()
    avg_time_cc = (end_cc - start_cc) / repeats

    # --- cv2.floodFill ---
    start_ff = time.perf_counter()
    for _ in range(repeats):
        filter_floodfill(image, max_size=200_000)
    end_ff = time.perf_counter()
    avg_time_ff = (end_ff - start_ff) / repeats

    # --- cv2.findContours ---
    start_fc = time.perf_counter()
    for _ in range(repeats):
        mask = cv2.compare(image, 0.5, cv2.CMP_GT)
        filter_contours(mask, max_size=200_000)
    end_fc = time.perf_counter()
    avg_time_fc = (end_fc - start_fc) / repeats

    # --- Results ---
    print(f"{sz} Average time over {repeats} runs:")
    print(f"  cv2.connectedComponentsWithStats: {avg_time_cc * 1e3:.3f} ms")
    print(f"  cv2.floodFill:                    {avg_time_ff * 1e3:.3f} ms, speedup: {avg_time_cc/avg_time_ff:.2f} X")
    print(f"  cv2.findContours:                 {avg_time_fc * 1e3:.3f} ms, speedup: {avg_time_cc/avg_time_fc:.2f} X")
