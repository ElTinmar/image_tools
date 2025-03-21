import numpy as np
import pytest
from image_tools.convert import im2single, im2uint8, im2half

@pytest.mark.benchmark
def test_uint8_to_float16_benchmark(benchmark):
    img = np.random.randint(0, 256, (2048,2048), dtype=np.uint8)  # Simulate a large image
    result = benchmark(im2half, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_float16_to_float16_benchmark(benchmark):
    img = np.random.random((2048,2048)).astype(np.float16)  # Simulate a large image
    result = benchmark(im2half, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_float32_to_float16_benchmark(benchmark):
    img = np.random.random((2048,2048)).astype(np.float32)  # Simulate a large image
    result = benchmark(im2half, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_uint8_to_float32_benchmark(benchmark):
    img = np.random.randint(0, 256, (2048,2048), dtype=np.uint8)  # Simulate a large image
    result = benchmark(im2single, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_float32_to_float32_benchmark(benchmark):
    img = np.random.random((2048,2048)).astype(np.float32)  # Simulate a large image
    result = benchmark(im2single, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_float64_to_float32_benchmark(benchmark):
    img = np.random.random((2048,2048)).astype(np.float64)  # Simulate a large image
    result = benchmark(im2single, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_bool_to_float32_benchmark(benchmark):
    img = np.random.random((2048,2048))>0.5  # Simulate a large image
    result = benchmark(im2single, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_uint8_to_uint8_benchmark(benchmark):
    img = np.random.randint(0, 256, (2048,2048), dtype=np.uint8)  # Simulate a large image
    result = benchmark(im2uint8, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_uint64_to_uint8_benchmark(benchmark):
    img = np.random.randint(0, 256, (2048,2048), dtype=np.uint64)  # Simulate a large image
    result = benchmark(im2uint8, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_float32_to_uint8_benchmark(benchmark):
    img = np.random.random((2048,2048)).astype(np.float32)  # Simulate a large image
    result = benchmark(im2uint8, img)
    assert result.shape == img.shape  # Ensure the function is correct

@pytest.mark.benchmark
def test_bool_to_uint8_benchmark(benchmark):
    img = np.random.random((2048,2048))>0.5  # Simulate a large image
    result = benchmark(im2uint8, img)
    assert result.shape == img.shape  # Ensure the function is correct