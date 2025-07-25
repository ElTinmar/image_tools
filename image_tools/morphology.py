import numpy as np
from scipy import ndimage as ndi
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Callable
from skimage.measure import regionprops
from skimage.measure._regionprops import _RegionProperties
import cv2
from dataclasses import dataclass
from geometry import pca

# those are essentially stripped down versions of 
# skimage.morphology.remove_small_objects

def label(        
        ar: NDArray, 
        connectivity: int = 1
    ) -> NDArray:

    strel = ndi.generate_binary_structure(ar.ndim, connectivity)
    return ndi.label(ar, strel)[0]

def properties(
        ar: NDArray, 
        connectivity: int = 1
    ) -> List[_RegionProperties]:

    label_img = label(ar, connectivity)
    return regionprops(label_img)

def components_size(
        ar: NDArray, 
        connectivity: int = 1
        ) -> Tuple[NDArray, NDArray]:
    
    ccs = label(ar, connectivity)
    component_sz = np.bincount(ccs.ravel()) 
    return (component_sz, ccs)

def bwareaopen(
        ar: NDArray, 
        min_size: int = 64, 
        connectivity: int = 1
        ) -> NDArray:
    
    out = ar.copy()
    component_sz, ccs = components_size(ar, connectivity)
    too_small = component_sz < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def bwareaclose(
        ar: NDArray, 
        max_size: int = 256, 
        connectivity: int = 1
        ) -> NDArray:
    
    out = ar.copy()
    component_sz, ccs = components_size(ar, connectivity)
    too_big = component_sz > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out

def bwareafilter(
        ar: NDArray, 
        min_size: int = 64, 
        max_size: int = 256, 
        connectivity: int = 1
        ) -> NDArray:
    
    out = ar.copy()
    component_sz, ccs = components_size(ar, connectivity)
    too_small = component_sz < min_size 
    too_small_mask = too_small[ccs]
    too_big = component_sz > max_size
    too_big_mask = too_big[ccs]
    out[too_small_mask] = 0
    out[too_big_mask] = 0

    return out

def bwareaopen_centroids(
        ar: NDArray, 
        min_size: int = 64,
        connectivity: int = 1,
    ) -> NDArray:
    
    props = properties(ar, connectivity)
    centroids = [blob.centroid[::-1] for blob in props if blob.area > min_size]
    return np.asarray(centroids, dtype=np.float32)

def bwareafilter_centroids(
        ar: NDArray, 
        min_size: int = 64, 
        max_size: int = 256, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 1
    ) -> NDArray:

    props = properties(ar, connectivity)
    centroids = []
    for blob in props:
        if not (min_size < blob.area < max_size):
            continue
        if (min_width is not None) and (max_width is not None) and (max_width > 0):
            if not (min_width < 2*blob.axis_minor_length < max_width):
                continue
        if (min_length is not None) and (max_length is not None)  and (max_length > 0):
            if not (min_length < 2*blob.axis_major_length < max_length):
                continue
        y, x = blob.centroid
        centroids.append([x, y])
    return np.asarray(centroids, dtype=np.float32)

def bwareaopen_props(
        ar: NDArray, 
        min_size: int = 64, 
        connectivity: int = 1
    ) -> List[_RegionProperties]:

    props = properties(ar, connectivity)
    return [blob for blob in props if blob.area > min_size]

def bwareafilter_props(
        ar: NDArray, 
        min_size: int = 64, 
        max_size: int = 256, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 1
    ) -> List[_RegionProperties]:

    props = properties(ar, connectivity)
    filtered_props = []
    for blob in props:
        if not (min_size < blob.area < max_size):
            continue
        if (min_width is not None) and (max_width is not None) and (max_width > 0):
            if not (min_width < 2*blob.axis_minor_length < max_width):
                continue
        if (min_length is not None) and (max_length is not None) and (max_length > 0):
            if not (min_length < 2*blob.axis_major_length < max_length):
                continue
        filtered_props.append(blob)
    return filtered_props

def bwareafilter_centroids_cv2(
        ar: cv2.UMat, 
        min_size: int = 64, 
        max_size: int = 256, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 8
    ) -> NDArray:
    # note that width and length are not treated equivalently to bwareafilter_centroids
    # here it is the width and height of the bounding box instead of minor and major axis
    # TODO handle None

    n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ar,
        connectivity = connectivity,
        ltype = cv2.CV_16U
    )
    kept_centroids = []
    for c in range(1, n_components):
        w = stats[c, cv2.CC_STAT_WIDTH]
        h = stats[c, cv2.CC_STAT_HEIGHT]
        area = stats[c, cv2.CC_STAT_AREA]
        keep_width = (max_width == 0) or (min_width is None) or (max_width is None) or (w > min_width and w < max_width)
        keep_height = (max_length == 0) or (min_length is None) or (max_length is None) or (h > min_length and h < max_length)
        keep_area = area > min_size and area < max_size
        if all((keep_width, keep_height, keep_area)):
            kept_centroids.append(centroids[c])

    return np.asarray(kept_centroids, dtype=np.float32)

@dataclass
class RegionPropsLike:
    centroid: NDArray # row, col
    coords: NDArray # row, col

    @property
    def principal_axis(self) -> Optional[NDArray]:
        up = np.array([0.0, 1.0])
        coords_xy = np.fliplr(self.coords) # row,col to x,y

        if (coords_xy.shape[0] <= 1) or np.any(np.var(coords_xy, axis=0) == 0):
            return None

        _, components, _ = pca(coords_xy) 
        principal_axis = components[:,0]

        # Resolve 180 deg ambiguity by aligning with up direction
        if principal_axis @ up < 0:
            principal_axis = -principal_axis

        return principal_axis

def bwareafilter_props_cv2(
        ar: cv2.UMat, 
        min_size: int = 64, 
        max_size: int = 256, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 8
    ) -> List[RegionPropsLike]:
    # return list of blobs, where blobs have centroid and coords

    n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ar,
        connectivity = connectivity,
        ltype = cv2.CV_16U
    )
    kept_blobs = []
    for c in range(1,n_components):
        w = stats[c, cv2.CC_STAT_WIDTH]
        h = stats[c, cv2.CC_STAT_HEIGHT]
        area = stats[c, cv2.CC_STAT_AREA]
        keep_width = (max_width == 0) or (min_width is None) or (max_width is None) or (w > min_width and w < max_width)
        keep_height = (max_length == 0) or (min_length is None) or (max_length is None) or (h > min_length and h < max_length)
        keep_area = area > min_size and area < max_size
        if all((keep_width, keep_height, keep_area)):
            blob = RegionPropsLike(
                centroid = centroids[c][::-1], # row, col
                coords = np.transpose(np.nonzero(labels == c)) # row, col
            )
            kept_blobs.append(blob)

    return kept_blobs

def bwareafilter_cv2(
        ar: cv2.UMat,
        min_size: int = 64, 
        max_size: int = 256, 
        connectivity: int = 8
        ) -> NDArray:
    
    n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ar,
        connectivity = connectivity,
        ltype = cv2.CV_16U
    )
    for c in range(1, n_components):
        area = stats[c, cv2.CC_STAT_AREA]
        keep_area = area > min_size and area < max_size
        if not keep_area:
            ar[labels == c] = 0

    return ar

@dataclass
class Blob:
    centroid: NDArray[np.float32]
    axes: NDArray[np.float32]
    width: float
    height: float
    area: float
    angle_rad: float

def filter_connected_comp_centroids(
        ar: cv2.UMat, 
        max_size: int,
        min_size: int = 0, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 8
    ) -> NDArray[np.float32]:

    if min_size < 0:
        raise ValueError('min_size must be positive')

    n_components, labels, stats, center = cv2.connectedComponentsWithStats(
        ar,
        connectivity = connectivity,
        ltype = cv2.CV_16U
    )
    
    centroids = []
    for c in range(1, n_components):
        width, height, area = stats[c, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA]]
        
        if not (min_size < area < max_size):
            continue
        if min_length is not None and height < min_length:
            continue
        if max_length is not None and height > max_length:
            continue
        if min_width is not None and width < min_width:
            continue
        if max_width is not None and width > max_width:
            continue
        
        centroids.append(center[c])

    return np.array(centroids, dtype=np.float32)

def filter_connected_comp(
        ar: cv2.UMat, 
        max_size: int = 256,
        min_size: int = 0, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        connectivity: int = 8
    ) -> List[Blob]: 

    if min_size < 0:
        raise ValueError('min_size must be positive')

    n_components, labels, stats, center = cv2.connectedComponentsWithStats(
        ar,
        connectivity = connectivity,
        ltype = cv2.CV_16U
    )

    blobs = []
    for c in range(1, n_components):

        left, top = stats[c, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP]]
        width, height = stats[c, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
        area = stats[c, cv2.CC_STAT_AREA]
        
        if not (min_size < area < max_size):
            continue
        if min_length is not None and height < min_length:
            continue
        if max_length is not None and height > max_length:
            continue
        if min_width is not None and width < min_width:
            continue
        if max_width is not None and width > max_width:
            continue
        
        sub_labels = labels[top: top+height, left:left+width]
        y, x = np.nonzero(sub_labels == c)
        coordinates = np.column_stack((left+x, top+y))
        mu, axes, scores = pca(coordinates)  

        if abs(max(scores[:,0])) > abs(min(scores[:,0])):
            axes[:,0] = -axes[:,0]

        if np.linalg.det(axes) < 0:
            axes[:,1] = -axes[:,1]

        blobs.append(
            Blob(
                centroid = mu,
                axes = axes,
                width = width,
                height = height,
                area = area,
                angle_rad = np.arctan2(axes[1,1], axes[0,1])
            )
        )

    return blobs

def filter_floodfill_centroid(
        ar: cv2.UMat, 
        max_size: int,
        min_size: int = 0, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        loDiff: float = 0.1,
        upDiff: float = 0.1, 
        seed_fun: Callable[[cv2.UMat], int] = np.argmax
    ) -> NDArray[np.float32]:

    # CAUTION: This returns only one centroid

    if min_size < 0:
        raise ValueError('min_size must be positive')

    mask = np.zeros((ar.shape[0]+2,ar.shape[1]+2), np.uint8)

    flat_index = seed_fun(ar)
    seed_y, seed_x = np.unravel_index(flat_index, ar.shape)

    area, _, mask, rect = cv2.floodFill(
        ar, 
        mask, 
        seedPoint = (seed_x, seed_y), 
        newVal = 255, 
        loDiff = loDiff,
        upDiff = upDiff , 
        flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    )

    left, top, width, height = rect

    if not (min_size < area < max_size):
        return np.zeros((0,2), dtype=np.float32)
    if min_length is not None and height < min_length:
        return np.zeros((0,2), dtype=np.float32)
    if max_length is not None and height > max_length:
        return np.zeros((0,2), dtype=np.float32)
    if min_width is not None and width < min_width:
        return np.zeros((0,2), dtype=np.float32)
    if max_width is not None and width > max_width:
        return np.zeros((0,2), dtype=np.float32)
    
    return np.array([[left+width//2, top+height//2]], dtype=np.float32)

def filter_floodfill(
        ar: cv2.UMat, 
        max_size: int = 256,
        min_size: int = 0, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        loDiff: float = 0.1,
        upDiff: float = 0.1, 
        seed_fun: Callable[[cv2.UMat], int] = np.argmax
    ) -> List[Blob]: 
    
    # CAUTION: This returns only one blob 

    if min_size < 0:
        raise ValueError('min_size must be positive')

    mask = np.zeros((ar.shape[0]+2,ar.shape[1]+2), np.uint8)

    flat_index = seed_fun(ar)
    seed_y, seed_x = np.unravel_index(flat_index, ar.shape)

    area, _, mask, rect = cv2.floodFill(
        ar, 
        mask, 
        seedPoint = (seed_x, seed_y), 
        newVal = 255, 
        loDiff = loDiff,
        upDiff = upDiff , 
        flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    )

    left, top, width, height = rect

    if not (min_size < area < max_size):
        return []
    if min_length is not None and height < min_length:
        return []
    if max_length is not None and height > max_length:
        return []
    if min_width is not None and width < min_width:
        return []
    if max_width is not None and width > max_width:
        return []

    sub_mask = mask[top+1:top+height+1, left+1:left+width+1]
    y, x = np.nonzero(sub_mask)
    coordinates = np.column_stack((left+x, top+y))
    mu, axes, scores = pca(coordinates)  

    if abs(max(scores[:,0])) > abs(min(scores[:,0])):
        axes[:,0] = -axes[:,0]

    if np.linalg.det(axes) < 0:
        axes[:,1] = -axes[:,1]

    blob = Blob(
        centroid = mu,
        axes = axes,
        width = width,
        height = height,
        area = area,
        angle_rad = np.arctan2(axes[1,1], axes[0,1])
    )

    return [blob]

def filter_contours_centroids(
        ar: cv2.UMat,
        max_size: int, 
        min_size: int = 0, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None
    ) -> NDArray[np.float32]:

    if min_size < 0:
        raise ValueError('min_size must be positive')
        
    contours, _ = cv2.findContours(ar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if not (min_size < area < max_size):
            continue
        
        (cx, cy), (width, height), angle = cv2.minAreaRect(cnt)
        if width < height:
            width, height = height, width
        
        if min_length is not None and height < min_length:
            continue
        if max_length is not None and height > max_length:
            continue
        if min_width is not None and width < min_width:
            continue
        if max_width is not None and width > max_width:
            continue

        centroids.append([cx, cy])
    
    return np.array(centroids, dtype=np.float32)

def filter_contours(
        ar: cv2.UMat,
        max_size: int, 
        min_size: int = 0, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None
    ) -> List[Blob]:

    if min_size < 0:
        raise ValueError('min_size must be positive')

    contours, _ = cv2.findContours(ar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if not (min_size < area < max_size):
            continue
        
        (cx, cy), (width, height), _ = cv2.minAreaRect(cnt)
        if width < height:
            width, height = height, width
        
        if min_length is not None and height < min_length:
            continue
        if max_length is not None and height > max_length:
            continue
        if min_width is not None and width < min_width:
            continue
        if max_width is not None and width > max_width:
            continue

        cnt_points = cnt.reshape(-1, 2)
        mass_center, axes, scores = pca(cnt_points)     
        
        if abs(max(scores[:,0])) > abs(min(scores[:,0])):
            axes[:,0] = -axes[:,0]

        if np.linalg.det(axes) < 0:
            axes[:,1] = -axes[:,1]

        blobs.append(
            Blob(
                centroid = mass_center,
                axes = axes,
                width = width,
                height = height,
                area = area,
                angle_rad = np.arctan2(axes[1,1], axes[0,1])
            )
        )

    return blobs