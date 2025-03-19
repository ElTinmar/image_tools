import numpy as np
import pytest
from image_tools.convert import im2single, im2uint8

@pytest.mark.benchmark
def test_uint8_to_float32_benchmark(benchmark):
    img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)  # Simulate a large image
    result = benchmark(im2single, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_float32_to_float32_benchmark(benchmark):
    img = np.random.random((1080, 1920, 3)).astype(np.float32)  # Simulate a large image
    result = benchmark(im2single, img)
    assert result.shape == img.shape  # Ensure the function is correct


@pytest.mark.benchmark
def test_float64_to_float32_benchmark(benchmark):
    img = np.random.random((1080, 1920, 3)).astype(np.float64)  # Simulate a large image
    result = benchmark(im2single, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_bool_to_float32_benchmark(benchmark):
    img = np.random.random((1080, 1920, 3))>0.5  # Simulate a large image
    result = benchmark(im2single, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_uint8_to_uint8_benchmark(benchmark):
    img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)  # Simulate a large image
    result = benchmark(im2uint8, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_uint64_to_uint8_benchmark(benchmark):
    img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint64)  # Simulate a large image
    result = benchmark(im2uint8, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_float32_to_uint8_benchmark(benchmark):
    img = np.random.random((1080, 1920, 3)).astype(np.float32)  # Simulate a large image
    result = benchmark(im2uint8, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_bool_to_uint8_benchmark(benchmark):
    img = np.random.random((1080, 1920, 3))>0.5  # Simulate a large image
    result = benchmark(im2uint8, img)
    assert result.shape == img.shape  # Ensure the function is correct