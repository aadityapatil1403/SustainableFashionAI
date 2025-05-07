import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import argparse
from shapely.geometry import Polygon
import glob

def extract_patterns_from_pdf(pdf_file, output_dir, min_contour_area=100):
    """
    Extract pattern pieces from a PDF file
    
    Args:
        pdf_file: Path to the PDF file
        output_dir: Directory to save extracted patterns
        min_contour_area: Minimum contour area to consider as a pattern piece
    
    Returns:
        List of pattern pieces
    """
    print(f"Processing {pdf_file}...")
    
    # Convert PDF to image
    images = convert_from_path(pdf_file, dpi=200)
    
    # Use the first page
    image = np.array(images[0])
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    print(f"Found {len(valid_contours)} pattern pieces")
    
    # Create visualization
    vis_image = np.ones_like(image) * 255
    cv2.drawContours(vis_image, valid_contours, -1, (0, 0, 255), 2)
    
    # Save visualization
    base_name = os.path.splitext(os.path.basename(pdf_file))[0]
    vis_file = os.path.join(output_dir, f"{base_name}_contours.png")
    cv2.imwrite(vis_file, vis_image)
    
    # Extract pattern pieces
    pattern_pieces = []
    for i, contour in enumerate(valid_contours):
        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to list of points
        points = approx.reshape(-1, 2).tolist()
        
        # Create pattern piece
        piece = {
            'id': f"{base_name}_piece_{i}",
            'points': points
        }
        
        pattern_pieces.append(piece)
    
    # Save to JSON
    json_file = os.path.join(output_dir, f"{base_name}.json")
    with open(json_file, 'w') as f:
        json.dump({'pattern_pieces': pattern_pieces}, f, indent=2)
    
    print(f"Saved {len(pattern_pieces)} pattern pieces to {json_file}")
    
    return pattern_pieces

def normalize_pattern_pieces(pattern_pieces, target_size=1000):
    """
    Normalize pattern pieces to fit within target_size
    """
    # Find min and max coordinates
    all_points = []
    for piece in pattern_pieces:
        all_points.extend(piece['points'])
    
    all_points = np.array(all_points)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    
    # Calculate scale factor
    width = max_x - min_x
    height = max_y - min_y
    scale = target_size / max(width, height)
    
    # Normalize points
    for piece in pattern_pieces:
        for i, point in enumerate(piece['points']):
            # Shift to origin and scale
            x = (point[0] - min_x) * scale
            y = (point[1] - min_y) * scale
            piece['points'][i] = [x, y]
    
    return pattern_pieces

def main():
    parser = argparse.ArgumentParser(description='Extract pattern pieces from PDF files')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='Path to input PDF files or directory')
    parser.add_argument('--output', type=str, default='hoodie_patterns',
                        help='Output directory')
    parser.add_argument('--min-area', type=int, default=1000,
                        help='Minimum contour area')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get input files
    input_files = []
    for input_path in args.input:
        if os.path.isdir(input_path):
            # If directory, get all PDF files
            pdf_files = glob.glob(os.path.join(input_path, '*.pdf'))
            input_files.extend(pdf_files)
        elif os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
            # If file, add to list
            input_files.append(input_path)
    
    if not input_files:
        print("No PDF files found")
        return
    
    print(f"Found {len(input_files)} PDF files")
    
    # Process each file
    all_pattern_pieces = []
    for pdf_file in input_files:
        pattern_pieces = extract_patterns_from_pdf(pdf_file, args.output, args.min_area)
        all_pattern_pieces.extend(pattern_pieces)
    
    # Normalize pattern pieces
    normalized_pieces = normalize_pattern_pieces(all_pattern_pieces)
    
    # Save all pattern pieces to a single file
    combined_file = os.path.join(args.output, 'all_patterns.json')
    with open(combined_file, 'w') as f:
        json.dump({'pattern_pieces': normalized_pieces}, f, indent=2)
    
    print(f"Saved {len(normalized_pieces)} normalized pattern pieces to {combined_file}")

if __name__ == "__main__":
    main()
