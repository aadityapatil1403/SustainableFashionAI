import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import svgwrite

def scan_document(image):
    """Simulate document scanning by improving contrast and removing shadows"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply bilateral filter to smooth the image while preserving edges
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding to handle varying illumination
    thresh = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Invert the image to get black lines on white background
    scanned = 255 - thresh
    
    return scanned

def preprocess_for_vectorizing(image):
    """Prepare image for vectorization with clean lines"""
    # Ensure image is binary with black lines on white background
    if np.mean(image) < 127:
        binary = 255 - image
    else:
        binary = image.copy()
    
    # Apply threshold to clean up gray areas
    _, binary = cv2.threshold(binary, 200, 255, cv2.THRESH_BINARY)
    
    # Invert for processing (white lines on black background)
    binary_inv = 255 - binary
    
    # Remove small dots and noise
    kernel = np.ones((2, 2), np.uint8)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
    
    # Thin lines for better tracing
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_ERODE, kernel, iterations=1)
    
    return binary_inv

def simplified_contours_to_svg(image, svg_path, width, height, min_path_length=20):
    """Create SVG with simplified paths to reduce number of objects"""
    drawing = svgwrite.Drawing(svg_path, size=(width, height))
    
    # Find contours with CV_RETR_LIST to get all contours in a flat list
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # Filter out tiny contours
    significant_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_path_length]
    
    # Group contours by proximity to reduce the number of paths
    grouped_contours = []
    current_group = []
    
    for contour in significant_contours:
        # Use Douglas-Peucker algorithm for path simplification
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to SVG path
        points = []
        for point in approx:
            x, y = point[0]
            points.append((x, y))
        
        if len(points) > 1:
            # Create path data
            path_data = f"M{points[0][0]},{points[0][1]} "
            for x, y in points[1:]:
                path_data += f"L{x},{y} "
            
            # Only close paths that are likely to be shapes
            arc_length = cv2.arcLength(contour, True)
            if arc_length > 100:
                path_data += "Z"
            
            # Add path to drawing with enhanced visibility (thicker stroke)
            drawing.add(drawing.path(d=path_data, fill="none", stroke="black", stroke_width=1.5))
    
    # Save SVG
    drawing.save()
    return svg_path

def process_sketches(input_folder, output_folder, display=True):
    """Process sketches to vectorized files with scanning preprocessing"""
    # Create output folders
    scanned_folder = os.path.join(output_folder, "scanned")
    vector_folder = os.path.join(output_folder, "vector")
    processed_folder = os.path.join(output_folder, "processed")
    
    Path(scanned_folder).mkdir(parents=True, exist_ok=True)
    Path(vector_folder).mkdir(parents=True, exist_ok=True)
    Path(processed_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Input folder: {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ No image files found in the input folder.")
        return
    
    print(f"\n🔍 Found {len(image_files)} images to process.")
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n✨ Processing image {idx}/{len(image_files)}: {image_file}")
        
        # Load image
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"❌ Error loading image: {image_file}")
            continue
        
        file_base = os.path.splitext(image_file)[0]
        height, width = image.shape[:2]
        
        # Process image to simulate scanning
        print("  🔄 Simulating document scanning...")
        scanned = scan_document(image)
        scanned_path = os.path.join(scanned_folder, f"{file_base}_scanned.png")
        cv2.imwrite(scanned_path, scanned)
        
        # Prepare for vectorization
        print("  🔄 Preparing for vectorization...")
        prepared = preprocess_for_vectorizing(scanned)
        processed_path = os.path.join(processed_folder, f"{file_base}_processed.png")
        cv2.imwrite(processed_path, prepared)
        
        # Convert to SVG with simplified contours
        print("  🔄 Creating simplified SVG...")
        svg_path = os.path.join(vector_folder, f"{file_base}.svg")
        svg_file = simplified_contours_to_svg(prepared, svg_path, width, height)
        
        print("  ✅ Successfully processed:")
        print(f"    📄 Scanned Image: {scanned_path}")
        print(f"    📄 Processed Image: {processed_path}")
        print(f"    📄 SVG Vector: {svg_path}")
        
        # Display images
        if display:
            plt.figure(figsize=(15, 8))
            
            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            
            plt.subplot(2, 2, 2)
            plt.title("Simulated Scan")
            plt.imshow(scanned, cmap="gray")
            plt.axis("off")
            
            plt.subplot(2, 2, 3)
            plt.title("Processed for Vectorization")
            plt.imshow(prepared, cmap="gray")
            plt.axis("off")
            
            # Draw contours on a blank image for visualization
            contour_img = np.zeros((height, width), dtype=np.uint8)
            contours, _ = cv2.findContours(prepared, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
            cv2.drawContours(contour_img, contours, -1, 255, 1)
            
            plt.subplot(2, 2, 4)
            plt.title("Simplified Contours")
            plt.imshow(contour_img, cmap="gray")
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
    
    print(f"\n🎉 All done! Processed {len(image_files)} images.")
    print(f"📁 Scanned images saved to: {scanned_folder}")
    print(f"📁 Processed images saved to: {processed_folder}")
    print(f"📁 Vector files saved to: {vector_folder}")
    print("\n💡 Next steps:")
    print("  1. Open the SVG files in Adobe Illustrator")
    print("  2. The SVGs should have fewer paths and be darker/more visible")
    print("  3. Use for tech pack generation")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Convert fashion sketches to vector files with document scanning')
    parser.add_argument('-i', '--input', type=str, help='Input folder containing sketch images')
    parser.add_argument('-o', '--output', type=str, help='Output folder for processed files')
    parser.add_argument('--no-display', action='store_true', help='Disable image display')
    parser.add_argument('--min-path', type=int, default=20, help='Minimum path length to include (filters out dots)')
    args = parser.parse_args()
    
    # Use command line arguments if provided, otherwise use defaults
    input_folder = args.input
    output_folder = args.output
    
    # If no input is provided, ask user
    if not input_folder:
        input_folder = input("📁 Enter input folder path: ")
    
    # If no output is provided, create a default
    if not output_folder:
        output_folder = os.path.join(os.path.dirname(input_folder), "vectorized_output")
        print(f"📁 Using default output folder: {output_folder}")
    
    # Validate input folder
    if not os.path.isdir(input_folder):
        print(f"❌ Input folder '{input_folder}' does not exist or is not a directory.")
        return
    
    # Process the sketches
    process_sketches(input_folder, output_folder, not args.no_display)

if __name__ == "__main__":
    print("🎨 Optimized Sketch to Vector Pipeline 🎨")
    print("--------------------------------------")
    main()