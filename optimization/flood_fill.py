import numpy as np
import cv2
import time

# Parameters
sizes = 2**np.arange(5,12)

for sz in sizes:

    print(sz)
    image_size = (sz, sz)
    disk_center = (sz//4, sz//6)
    disk_radius = sz//12
    repeats = 500
    
    # Create binary image with a white disk
    image = np.zeros(image_size, dtype=np.float32)
    cv2.circle(image, disk_center, disk_radius, 1, -1)
            
    # --- cv2.connectedComponentsWithStats ---
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

    # --- cv2.floodFill ---
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

    # --- cv2.moments ---
    start_mm = time.perf_counter()
    for _ in range(repeats):
        mask = cv2.compare(image, 0.5, cv2.CMP_GT)
        m2 = cv2.moments(mask, binaryImage=True)
        centroid = m2['m10']/m2['m00'], m2['m01']/m2['m00']
    end_mm = time.perf_counter()
    avg_time_mm = (end_mm - start_mm) / repeats

    # --- cv2.findContours ---
    start_fc = time.perf_counter()
    for _ in range(repeats):
        mask = cv2.compare(image, 0.5, cv2.CMP_GT)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                centroids.append((M["m10"]/M["m00"], M["m01"]/M["m00"]))
    end_fc = time.perf_counter()
    avg_time_fc = (end_fc - start_fc) / repeats

    # --- Results ---
    print(f"Average time over {repeats} runs:")
    print(f"  cv2.connectedComponentsWithStats: {avg_time_cc * 1e3:.3f} ms")
    print(f"  cv2.floodFill:                    {avg_time_ff * 1e3:.3f} ms, speedup: {avg_time_cc/avg_time_ff:.2f} X")
    print(f"  cv2.moments:                      {avg_time_mm * 1e3:.3f} ms, speedup: {avg_time_cc/avg_time_mm:.2f} X")
    print(f"  cv2.findContours:                 {avg_time_fc * 1e3:.3f} ms, speedup: {avg_time_cc/avg_time_fc:.2f} X")
