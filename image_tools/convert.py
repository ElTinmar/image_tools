import numpy as np
from numpy.typing import NDArray

def im2half(input_image: NDArray) -> NDArray:
    """
    Transform input image into a float16 precision floating point image.
    Note that this is slow for large images
    """

    if input_image.dtype == np.float16:
        return input_image    
    
    if np.issubdtype(input_image.dtype, np.integer):
        # if integer type, transform to float and scale between 0 and 1
        out = np.empty_like(input_image, dtype=np.float16)
        np.divide(input_image, np.iinfo(input_image.dtype).max, out=out)
        return out

    if np.issubdtype(input_image.dtype, np.floating) or np.issubdtype(input_image.dtype, np.bool_):
        return input_image.astype(np.float16)

    raise ValueError('wrong image type, cannot convert to single')

def im2single(input_image: NDArray) -> NDArray:
    """
    Transform input image into a single precision floating point image.
    Note that this is slow for large images
    """

    if input_image.dtype == np.float32:
        return input_image    
    
    if np.issubdtype(input_image.dtype, np.integer):
        # if integer type, transform to float and scale between 0 and 1
        out = np.empty_like(input_image, dtype=np.float32)
        np.divide(input_image, np.iinfo(input_image.dtype).max, out=out, dtype=np.float32)
        return out

    if np.issubdtype(input_image.dtype, np.floating) or np.issubdtype(input_image.dtype, np.bool_):
        return input_image.astype(np.float32)

    raise ValueError('wrong image type, cannot convert to single')

def im2double(input_image: NDArray) -> NDArray:
    """
    Transform input image into a double precision floating point image
    """

    if np.issubdtype(input_image.dtype, np.integer):
        # if integer type, transform to float and scale between 0 and 1
        ui_info = np.iinfo(input_image.dtype)
        double_image = input_image.astype(np.float64) / ui_info.max

    elif np.issubdtype(input_image.dtype, np.floating):
        # if already a floating type, convert to double precision
        if input_image.dtype == np.float64:
            return input_image
        
        double_image = input_image.astype(np.float64)

    elif input_image.dtype == np.bool_:
        double_image = input_image.astype(np.float64) 

    else:
        raise ValueError('wrong image type, cannot convert to double')
    
    return double_image

def im2uint8(input_image: NDArray) -> NDArray:
    '''Convert image to uint8. Note that this is slow for large images'''
    
    if input_image.dtype == np.uint8:
        return input_image   

    out = np.empty_like(input_image, dtype=np.uint8)

    if np.issubdtype(input_image.dtype, np.integer):
        np.multiply(
            input_image, 
            255.0 / np.iinfo(input_image.dtype).max, 
            out=out, 
            casting='unsafe'
        )
        return out

    if np.issubdtype(input_image.dtype, np.floating):
        np.multiply(input_image, 255.0, out=out, casting='unsafe')
        return out

    if input_image.dtype == np.bool_:
        np.multiply(input_image, 255, out=out, dtype=np.uint8)
        return out

    raise ValueError('wrong image type, cannot convert to uint8')

def rgb2gray(input_image: NDArray) -> NDArray:
    """
    Transform color input into grayscale by taking only the first channel

    Inputs:
        input_image: M x N x C | M x N x C x K numpy array 

    Outputs:
        M x N | M x N x K numpy array 
    """

    shp = input_image.shape

    if len(shp) == 2:
        # already grayscale, nothing to do
        return input_image
    
    if len(shp) >= 3:
        # M x N X C
        return np.dot(input_image[...,:3], np.array([0.2990, 0.5870, 0.1140], dtype=np.float32))
    
    else:
        raise ValueError('wrong image type, cannot convert to grayscale')

def im2gray(input_image: NDArray) -> NDArray:
    """
    Transform color input into grayscale by taking only the first channel

    Inputs:
        input_image: M x N x C | M x N x C x K numpy array 

    Outputs:
        M x N | M x N x K numpy array 
    """

    shp = input_image.shape

    if len(shp) == 2:
        # already grayscale, nothing to do
        return input_image
    
    if len(shp) >= 3:
        return input_image[...,0]
    
    else:
        raise ValueError('wrong image type, cannot convert to grayscale')
    
def im2rgb(input_image: NDArray) -> NDArray:
    """
    Transform grayscale input into color image
    """

    shp = input_image.shape

    if len(shp) == 3 and shp[2] == 3:
        # already RGB, nothing to do
        return input_image
    
    elif len(shp) == 2:
        rgb_image = np.dstack((input_image, input_image, input_image))
        return rgb_image
    
    else:
        raise ValueError('wrong image type, cannot convert to RGB')
    
