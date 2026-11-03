
import cv2
import numpy as np


def preprocess_prescription_image(image_bytes: bytes) -> bytes:
    """
    Enhance a prescription image for OCR and vision models.
    Returns an enhanced PNG image as bytes.
    """

    if not image_bytes:
        raise ValueError("Image is empty.")

    # Decode uploaded image
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode prescription image.")

    # Enlarge the image 2x using high-quality interpolation
    enlarged = cv2.resize(
        image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Convert to grayscale
    grayscale = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(grayscale)

    # Reduce noise while preserving handwriting edges
    denoised = cv2.bilateralFilter(
        enhanced,
        5,
        40,
        40
    )

    # Sharpen the handwriting
    blurred = cv2.GaussianBlur(
        denoised,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        denoised,
        1.5,
        blurred,
        -0.5,
        0
    )

    # Encode as PNG
    success, encoded = cv2.imencode(
        ".png",
        sharpened
    )

    if not success:
        raise ValueError("Could not encode enhanced image.")

    return encoded.tobytes()
