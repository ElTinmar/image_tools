import cProfile
import pstats

from .morphology import filter_contours, filter_contours_centroids, bwareafilter_props_cv2
import numpy as np
import cv2

sz = 2048
image_size = (sz, sz)
disk_center = (sz//4, sz//6)
disk_radius = sz//12
image = np.zeros(image_size, dtype=np.float32)
cv2.circle(image, disk_center, disk_radius, 1, -1)
mask = cv2.compare(image, 0.5, cv2.CMP_GT)
repeats = 500

with cProfile.Profile() as pr:
    for _ in range(repeats):
        filter_contours(mask, min_size=0, max_size=200_000)
        ps = pstats.Stats(pr)
        ps.dump_stats('filter_contours.prof')