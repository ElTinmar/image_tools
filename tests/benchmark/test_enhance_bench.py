import numpy as np
import pytest
from image_tools import enhance  

@pytest.fixture
def test_image():
    """Generate a random grayscale image."""
    return np.random.rand(2048, 2048).astype(np.float32)  # Normalized 0-1

def test_enhance_functionality(test_image):
    """Test if enhance function behaves correctly."""
    output = enhance(test_image, contrast=1.2, gamma=1.1, brightness=0.05, blur_size_px=3, medfilt_size_px=3)

    # Check shape & dtype
    assert output.shape == test_image.shape
    assert output.dtype == np.float32

    # Ensure values are within the valid range
    assert np.all(output >= 0) and np.all(output <= 1)


def test_benchmark_enhance_no_filters(benchmark, test_image):
    """Benchmark enhance function without blur or median filtering."""
    benchmark(enhance, test_image, contrast=1.2, gamma=1.1, brightness=0.05)


def test_benchmark_enhance_with_blur(benchmark, test_image):
    """Benchmark enhance function with blur applied."""
    benchmark(enhance, test_image, contrast=1.2, gamma=1.1, brightness=0.05, blur_size_px=3)


def test_benchmark_enhance_with_medfilt(benchmark, test_image):
    """Benchmark enhance function with median filtering."""
    benchmark(enhance, test_image, contrast=1.2, gamma=1.1, brightness=0.05, medfilt_size_px=3)


def test_benchmark_enhance_with_all_filters(benchmark, test_image):
    """Benchmark enhance function with both blur and median filtering."""
    benchmark(enhance, test_image, contrast=1.2, gamma=1.1, brightness=0.05, blur_size_px=3, medfilt_size_px=3)
