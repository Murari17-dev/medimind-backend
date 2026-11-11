from io import BytesIO

import pytest
from PIL import Image, UnidentifiedImageError

from image_preprocessor import preprocess_prescription_image
from parser import preprocess_image


def make_png(width=12, height=8):
    output = BytesIO()
    Image.new("RGB", (width, height), "white").save(
        output,
        format="PNG",
    )
    return output.getvalue()


def test_opencv_preprocessor_returns_enlarged_png():
    result = preprocess_prescription_image(make_png())

    with Image.open(BytesIO(result)) as image:
        assert image.format == "PNG"
        assert image.size == (24, 16)


@pytest.mark.parametrize("image_bytes", [b"", b"not an image"])
def test_opencv_preprocessor_rejects_invalid_input(image_bytes):
    with pytest.raises(ValueError):
        preprocess_prescription_image(image_bytes)


def test_production_preprocessor_returns_jpeg():
    result = preprocess_image(make_png())

    with Image.open(BytesIO(result)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (12, 8)


def test_production_preprocessor_limits_large_dimensions():
    result = preprocess_image(make_png(width=2201, height=10))

    with Image.open(BytesIO(result)) as image:
        assert max(image.size) <= 2200


def test_production_preprocessor_rejects_invalid_image():
    with pytest.raises(UnidentifiedImageError):
        preprocess_image(b"not an image")
