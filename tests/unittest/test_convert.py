import numpy as np
import pytest
from image_tools import (
    im2half, im2single, im2double, im2uint8,
    rgb2gray, im2gray, im2rgb
)

@pytest.fixture
def sample_images():
    """Provides sample images of various types for testing."""
    return {
        "uint8": np.array([[0, 128, 255], [34, 67, 200]], dtype=np.uint8),
        "uint16": np.array([[0, 32768, 65535], [1234, 5678, 30000]], dtype=np.uint16),
        "float16": np.array([[0.0, 0.5, 1.0], [0.2, 0.7, 0.9]], dtype=np.float16),
        "float32": np.array([[0.0, 0.5, 1.0], [0.2, 0.7, 0.9]], dtype=np.float32),
        "float64": np.array([[0.0, 0.5, 1.0], [0.2, 0.7, 0.9]], dtype=np.float64),
        "bool": np.array([[True, False, True], [False, True, False]], dtype=bool),
        "rgb": np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8),
        "gray": np.random.randint(0, 255, (3, 3), dtype=np.uint8),
    }

def test_im2half_bool(sample_images):
    img = sample_images["bool"]
    converted = im2half(img)
    assert converted.dtype == np.float16
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2half_uint8(sample_images):
    img = sample_images["uint8"]
    converted = im2half(img)
    assert converted.dtype == np.float16
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2half_uint16(sample_images):
    img = sample_images["uint16"]
    converted = im2half(img)
    assert converted.dtype == np.float16
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2half_float16(sample_images):
    img = sample_images["float16"]
    converted = im2half(img)
    assert converted.dtype == np.float16
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2half_float32(sample_images):
    img = sample_images["float32"]
    converted = im2half(img)
    assert converted.dtype == np.float16
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2half_float64(sample_images):
    img = sample_images["float64"]
    converted = im2half(img)
    assert converted.dtype == np.float16
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2single_bool(sample_images):
    img = sample_images["bool"]
    converted = im2single(img)
    assert converted.dtype == np.float32
    assert np.array_equal(converted, img.astype(np.float32))

def test_im2single_uint8(sample_images):
    img = sample_images["uint8"]
    converted = im2single(img)
    assert converted.dtype == np.float32
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2single_uint16(sample_images):
    img = sample_images["uint16"]
    converted = im2single(img)
    assert converted.dtype == np.float32
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2single_float16(sample_images):
    img = sample_images["float16"]
    converted = im2single(img)
    assert converted.dtype == np.float32
    assert np.array_equal(converted, img.astype(np.float32))

def test_im2single_float32(sample_images):
    img = sample_images["float32"]
    converted = im2single(img)
    assert converted.dtype == np.float32
    assert np.array_equal(converted, img.astype(np.float32))

def test_im2single_float64(sample_images):
    img = sample_images["float64"]
    converted = im2single(img)
    assert converted.dtype == np.float32
    assert np.array_equal(converted, img.astype(np.float32))

def test_im2double_bool(sample_images):
    img = sample_images["bool"]
    converted = im2double(img)
    assert converted.dtype == np.float64
    assert np.array_equal(converted, img.astype(np.float64))

def test_im2double_uint8(sample_images):
    img = sample_images["uint8"]
    converted = im2double(img)
    assert converted.dtype == np.float64
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2double_uint16(sample_images):
    img = sample_images["uint16"]
    converted = im2double(img)
    assert converted.dtype == np.float64
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2double_float16(sample_images):
    img = sample_images["float16"]
    converted = im2double(img)
    assert converted.dtype == np.float64
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2double_float32(sample_images):
    img = sample_images["float32"]
    converted = im2double(img)
    assert converted.dtype == np.float64
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2double_float64(sample_images):
    img = sample_images["float64"]
    converted = im2double(img)
    assert converted.dtype == np.float64
    assert np.all((converted >= 0) & (converted <= 1))

def test_im2uint8_bool(sample_images):
    img = sample_images["bool"]
    converted = im2uint8(img)
    assert converted.dtype == np.uint8
    assert np.all((converted == 0) | (converted == 255))

def test_im2uint8_uint8(sample_images):
    img = sample_images["uint8"]
    converted = im2uint8(img)
    assert converted.dtype == np.uint8
    assert np.all((converted >= 0) & (converted <= 255))

def test_im2uint8_uint16(sample_images):
    img = sample_images["uint16"]
    converted = im2uint8(img)
    assert converted.dtype == np.uint8
    assert np.all((converted >= 0) & (converted <= 255))

def test_im2uint8_float16(sample_images):
    img = sample_images["float16"]
    converted = im2uint8(img)
    assert converted.dtype == np.uint8
    assert np.all((converted >= 0) & (converted <= 255))

def test_im2uint8_float32(sample_images):
    img = sample_images["float32"]
    converted = im2uint8(img)
    assert converted.dtype == np.uint8
    assert np.all((converted >= 0) & (converted <= 255))

def test_im2uint8_float64(sample_images):
    img = sample_images["float64"]
    converted = im2uint8(img)
    assert converted.dtype == np.uint8
    assert np.all((converted >= 0) & (converted <= 255))

def test_rgb2gray(sample_images):
    img = sample_images["rgb"].astype(np.float32) / 255.0  # Normalize
    gray = rgb2gray(img)
    assert gray.shape == img.shape[:2]  # Should remove the color dimension
    assert gray.dtype == np.float32  # Should maintain float type

def test_im2gray(sample_images):
    img = sample_images["rgb"]
    gray = im2gray(img)
    assert gray.shape == img.shape[:2]  # Should remove the color dimension
    assert gray.dtype == img.dtype  # Should maintain the same type

def test_im2rgb(sample_images):
    img = sample_images["gray"]
    rgb = im2rgb(img)
    assert rgb.shape == (img.shape[0], img.shape[1], 3)  # Should add color dimension
    assert rgb.dtype == img.dtype  # Should maintain the same type

def test_invalid_rgb2gray():
    with pytest.raises(ValueError):
        rgb2gray(np.array([1, 2, 3]))  # Not a valid image shape

def test_invalid_im2rgb():
    with pytest.raises(ValueError):
        im2rgb(np.array([1, 2, 3]))  # Not a valid image shape

