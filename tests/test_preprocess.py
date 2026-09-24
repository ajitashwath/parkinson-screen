from __future__ import annotations

import numpy as np

from common.preprocess import (
    PreprocessConfig,
    crop_to_content,
    load_grayscale,
    normalize,
    preprocess_image,
    resize_with_padding,
    skeleton_rerender,
    to_three_channel,
)


def test_load_grayscale_from_pil_image(spiral_image_factory) -> None:
    image = spiral_image_factory()
    gray = load_grayscale(image)
    assert gray.dtype == np.uint8
    assert gray.shape == (256, 256)


def test_load_grayscale_from_bytes(spiral_image_factory) -> None:
    import io

    image = spiral_image_factory()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    gray = load_grayscale(buffer.getvalue())
    assert gray.shape == (256, 256)


def test_load_grayscale_from_path(tmp_path, spiral_image_factory) -> None:
    path = tmp_path / "spiral.png"
    spiral_image_factory().save(path)
    gray = load_grayscale(path)
    assert gray.shape == (256, 256)


def test_crop_to_content_shrinks_blank_border(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    cropped = crop_to_content(gray, padding_frac=0.05)
    assert cropped.shape[0] < gray.shape[0]
    assert cropped.shape[1] < gray.shape[1]
    assert (cropped < 200).any()


def test_crop_to_content_handles_blank_image(blank_image) -> None:
    gray = np.array(blank_image)
    cropped = crop_to_content(gray)
    assert np.array_equal(cropped, gray)


def test_resize_with_padding_returns_exact_square(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory(size=300))
    resized = resize_with_padding(gray, size=224)
    assert resized.shape == (224, 224)


def test_resize_with_padding_pads_non_square_input(spiral_image_factory) -> None:
    tall = np.full((300, 150), 255, dtype=np.uint8)
    tall[50:250, 60:90] = 0
    resized = resize_with_padding(tall, size=224)
    assert resized.shape == (224, 224)
    assert (resized[:, 0] == 255).all()
    assert (resized[:, -1] == 255).all()


def test_normalize_scales_to_unit_range() -> None:
    gray = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    normalized = normalize(gray)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized[0, 1] == 1.0
    assert normalized[0, 0] == 0.0


def test_to_three_channel_replicates() -> None:
    array = np.zeros((10, 10), dtype=np.float32)
    array[0, 0] = 0.5
    three = to_three_channel(array)
    assert three.shape == (10, 10, 3)
    assert np.array_equal(three[..., 0], three[..., 1])
    assert np.array_equal(three[..., 0], three[..., 2])


def test_preprocess_image_output_shape_and_range(spiral_image_factory) -> None:
    output = preprocess_image(spiral_image_factory(), PreprocessConfig(size=128))
    assert output.shape == (128, 128, 3)
    assert output.dtype == np.float32
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_preprocess_image_is_deterministic(spiral_image_factory) -> None:
    image = spiral_image_factory()
    first = preprocess_image(image)
    second = preprocess_image(image)
    assert np.array_equal(first, second)


def test_preprocess_image_handles_blank_input(blank_image) -> None:
    output = preprocess_image(blank_image, PreprocessConfig(size=64))
    assert output.shape == (64, 64, 3)
    assert np.isclose(output.max(), 1.0)


def test_skeleton_rerender_mode_produces_thin_strokes(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    rerendered = skeleton_rerender(gray, stroke_width=3)
    original_stroke_pixels = int((gray < 200).sum())
    rerendered_stroke_pixels = int((rerendered < 200).sum())
    assert rerendered_stroke_pixels > 0
    assert rerendered_stroke_pixels < original_stroke_pixels * 3


def test_skeleton_rerender_mode_via_preprocess_image(spiral_image_factory) -> None:
    config = PreprocessConfig(size=128, mode="skeleton_rerender")
    output = preprocess_image(spiral_image_factory(), config)
    assert output.shape == (128, 128, 3)


def test_load_grayscale_from_rgb_array(spiral_image_factory) -> None:
    rgb = np.array(spiral_image_factory().convert("RGB"))
    gray = load_grayscale(rgb)
    assert gray.shape == (256, 256)
    assert gray.dtype == np.uint8


def test_load_grayscale_passthrough_for_2d_array() -> None:
    array = np.full((20, 20), 100, dtype=np.uint8)
    gray = load_grayscale(array)
    assert np.array_equal(gray, array)


def test_preprocess_image_accepts_ndarray_source(spiral_image_factory) -> None:
    array = np.array(spiral_image_factory())
    output = preprocess_image(array, PreprocessConfig(size=64))
    assert output.shape == (64, 64, 3)
