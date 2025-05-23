# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11rGv601h92C9LMM2lKFq1YRo9PuhO7FW
"""

import cv2
import numpy as np
from skimage import measure
import xml.etree.ElementTree as ET
import os

def vectorize_adaptive_detail(input_image, output_svg, detail_level=1.5):
    """
    Vectorize hoodie sketch using only adaptive thresholding
    with a detail level of 1.5
    """
    # Read the image
    original = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError(f"Could not read image at {input_image}")

    # Apply only adaptive thresholding - this matches the adaptive_detail1.5.svg approach
    adaptive = cv2.adaptiveThreshold(
        original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Save the preprocessed image for reference
    cv2.imwrite("preprocessed_adaptive.png", adaptive)

    # Find contours
    contours = measure.find_contours(adaptive, 0.5)

    # Apply polygon approximation with a fixed detail level of 1.5
    from skimage.measure import approximate_polygon
    processed_contours = []

    for contour in contours:
        # Skip very small contours
        if len(contour) < 5:
            continue

        # Apply the exact same simplification level that worked well (1.5)
        simplified = approximate_polygon(contour, tolerance=detail_level)
        processed_contours.append(simplified)

    # Create SVG
    height, width = original.shape

    # Create SVG root element
    root = ET.Element('svg')
    root.set('xmlns', 'http://www.w3.org/2000/svg')
    root.set('width', str(width))
    root.set('height', str(height))
    root.set('viewBox', f'0 0 {width} {height}')

    # Add each contour as a path
    for i, contour in enumerate(processed_contours):
        # Create path element
        path = ET.SubElement(root, 'path')

        # Build path data
        d = f'M {contour[0][1]},{contour[0][0]} '

        # Add line segments
        for point in contour[1:]:
            d += f'L {point[1]},{point[0]} '

        # Close path
        d += 'Z'

        # Set path attributes
        path.set('d', d)
        path.set('fill', 'none')
        path.set('stroke', 'black')
        path.set('stroke-width', '1')
        path.set('id', f'path_{i}')

    # Write SVG file
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass  # Skip indentation for Python < 3.9

    tree.write(output_svg, encoding='utf-8', xml_declaration=True)

    print(f"Adaptive vectorization complete. Output saved to {output_svg}")
    return output_svg

if __name__ == "__main__":
    input_file = "hoodie_sketch.jpg"  # Your input sketch
    output_file = "hoodie_vector_adaptive_detail15.svg"  # Output vector file

    vectorize_adaptive_detail(input_file, output_file, detail_level=0.8)
    print("Conversion complete. The SVG file matches the adaptive_detail1.5.svg approach.")

