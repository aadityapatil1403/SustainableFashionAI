import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import svgwrite
from PIL import Image
import io
import subprocess
import sys
import platform

def load_image(image_path):
    """Load image using Pillow first, falling back to OpenCV"""
    file_ext = os.path.splitext(image_path)[1].lower()
    
    try:
        # Try to open with Pillow (handles most formats)
        img = Image.open(image_path)
        # Convert to numpy array for OpenCV processing
        img_array = np.array(img)
        # Convert RGB to BGR for OpenCV compatibility
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array
    except Exception as e:
        print(f"Pillow couldn't open the image, trying OpenCV: {e}")
        # Fall back to OpenCV
        return cv2.imread(image_path)

def enhance_sketch(image):
    """Enhance sketch to make lines clearer and more defined"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Adjust brightness and contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0     # Brightness control (0-100)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    # Denoise
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    return enhanced

def process_for_vectorization(image):
    """Process image specifically for better vectorization results"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Sharpen the image to enhance edges
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
    
    # Binarize with adaptive threshold
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Clean small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Close small gaps in lines
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def create_svg_for_techpack(image, svg_path, width, height, min_contour_length=15, connect_gaps=True):
    """Create SVG optimized for tech pack use with connected paths and simplified structure"""
    drawing = svgwrite.Drawing(svg_path, size=(width, height))
    
    # Find all contours - use external and internal for complete details
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # Skip tiny contours
    significant_contours = []
    for i, contour in enumerate(contours):
        length = cv2.arcLength(contour, True)
        if length >= min_contour_length:
            significant_contours.append((contour, length))
    
    # Sort by length (largest first)
    significant_contours.sort(key=lambda x: x[1], reverse=True)
    
    # Process and add paths to SVG
    for contour, length in significant_contours:
        # Apply Douglas-Peucker algorithm for smoother curves
        # Smaller epsilon = more detail, larger = more smoothing
        epsilon = 0.001 * length
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to path points
        points = []
        for point in approx:
            x, y = point[0]
            points.append((x, y))
        
        if len(points) > 1:
            # Determine stroke width based on contour length
            # Important contours (like outlines) get thicker lines
            if length > 200:  # Main garment outlines
                stroke_width = 2.0
            elif length > 100:  # Important details
                stroke_width = 1.5
            else:  # Fine details
                stroke_width = 1.0
            
            # Create path data
            path_data = f"M{points[0][0]},{points[0][1]} "
            for x, y in points[1:]:
                path_data += f"L{x},{y} "
            
            # Close path for shapes but not for lines
            if cv2.contourArea(contour) > 100:
                path_data += "Z"
            
            # Add path to SVG
            drawing.add(drawing.path(d=path_data, fill="none", stroke="black", stroke_width=stroke_width))
    
    # Save the SVG
    drawing.save()
    return svg_path

def process_sketches(input_folder, output_folder, display=True, min_path_length=15):
    """Process sketches to tech-pack ready vector files"""
    # Create output folders
    enhanced_folder = os.path.join(output_folder, "enhanced")
    vector_folder = os.path.join(output_folder, "vector")
    processed_folder = os.path.join(output_folder, "processed")
    
    Path(enhanced_folder).mkdir(parents=True, exist_ok=True)
    Path(vector_folder).mkdir(parents=True, exist_ok=True)
    Path(processed_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Input folder: {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif'))]
    
    if not image_files:
        print("❌ No image files found in the input folder.")
        return
    
    print(f"\n🔍 Found {len(image_files)} images to process.")
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n✨ Processing image {idx}/{len(image_files)}: {image_file}")
        
        # Load image
        image_path = os.path.join(input_folder, image_file)
        image = load_image(image_path)
        
        if image is None:
            print(f"❌ Error loading image: {image_file}")
            continue
        
        file_base = os.path.splitext(image_file)[0]
        height, width = image.shape[:2]
        
        # Enhance sketch for better line clarity
        print("  🔄 Enhancing sketch quality...")
        enhanced = enhance_sketch(image)
        enhanced_path = os.path.join(enhanced_folder, f"{file_base}_enhanced.png")
        cv2.imwrite(enhanced_path, enhanced)
        
        # Process for vectorization
        print("  🔄 Processing for clean vectorization...")
        processed = process_for_vectorization(enhanced)
        processed_path = os.path.join(processed_folder, f"{file_base}_processed.png")
        cv2.imwrite(processed_path, processed)
        
        # Create tech-pack ready SVG
        print("  🔄 Creating tech-pack ready SVG...")
        svg_path = os.path.join(vector_folder, f"{file_base}.svg")
        create_svg_for_techpack(processed, svg_path, width, height, min_path_length)
        
        print("  ✅ Successfully processed:")
        print(f"    📄 Enhanced Image: {enhanced_path}")
        print(f"    📄 Processed Image: {processed_path}")
        print(f"    📄 Tech Pack SVG: {svg_path}")
        
        # Display results if requested
        if display:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.title("Original Sketch")
            if len(image.shape) == 3:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(image, cmap="gray")
            plt.axis("off")
            
            plt.subplot(2, 2, 2)
            plt.title("Enhanced Sketch")
            plt.imshow(enhanced, cmap="gray")
            plt.axis("off")
            
            plt.subplot(2, 2, 3)
            plt.title("Processing for Vectorization")
            plt.imshow(processed, cmap="gray")
            plt.axis("off")
            
            # Create a visualization of the contours
            contour_img = np.zeros((height, width, 3), dtype=np.uint8)
            contour_img.fill(255)  # White background
            
            contours, _ = cv2.findContours(processed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
            cv2.drawContours(contour_img, contours, -1, (0, 0, 0), 1)  # Black contours
            
            plt.subplot(2, 2, 4)
            plt.title("Final Contours")
            plt.imshow(contour_img)
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
    
    print(f"\n🎉 All done! Processed {len(image_files)} images.")
    print(f"📁 Enhanced images saved to: {enhanced_folder}")
    print(f"📁 Processed images saved to: {processed_folder}")
    print(f"📁 Tech Pack SVGs saved to: {vector_folder}")
    print("\n💡 Next steps for your tech pack workflow:")
    print("  1. Open the SVG files in Adobe Illustrator")
    print("  2. Use the Pen tool to connect any remaining gaps")
    print("  3. Use the Live Paint Bucket to fill enclosed areas")
    print("  4. Add measurements and annotations for the tech pack")

def open_illustrator_with_file(file_path):
    """Attempt to open a file with Adobe Illustrator"""
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.call(['open', '-a', 'Adobe Illustrator', file_path])
            return True
        elif platform.system() == 'Windows':
            # Try common installation paths
            illustrator_paths = [
                r"C:\Program Files\Adobe\Adobe Illustrator 2023\Support Files\Contents\Windows\Illustrator.exe",
                r"C:\Program Files\Adobe\Adobe Illustrator 2022\Support Files\Contents\Windows\Illustrator.exe",
                r"C:\Program Files\Adobe\Adobe Illustrator 2021\Support Files\Contents\Windows\Illustrator.exe",
                # Add more potential paths as needed
            ]
            
            for ai_path in illustrator_paths:
                if os.path.exists(ai_path):
                    subprocess.Popen([ai_path, file_path])
                    return True
            
            # If specific paths don't work, try with the file association
            os.startfile(file_path)
            return True
        
        elif platform.system() == 'Linux':
            subprocess.call(['xdg-open', file_path])
            return True
        
    except Exception as e:
        print(f"⚠️ Could not open file with Illustrator: {e}")
        return False

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Convert fashion sketches to tech pack ready vector files')
    parser.add_argument('-i', '--input', type=str, help='Input folder containing sketch images')
    parser.add_argument('-o', '--output', type=str, help='Output folder for processed files')
    parser.add_argument('--no-display', action='store_true', help='Disable image display')
    parser.add_argument('--min-path', type=int, default=15, help='Minimum path length to include (filters out dots)')
    parser.add_argument('--open-with-ai', action='store_true', help='Try to open resulting SVG with Adobe Illustrator')
    args = parser.parse_args()
    
    # Use command line arguments if provided, otherwise use defaults
    input_folder = args.input
    output_folder = args.output
    min_path_length = args.min_path
    
    # If no input is provided, ask user
    if not input_folder:
        input_folder = input("📁 Enter input folder path: ")
    
    # If no output is provided, create a default
    if not output_folder:
        output_folder = os.path.join(os.path.dirname(input_folder), "techpack_ready")
        print(f"📁 Using default output folder: {output_folder}")
    
    # Validate input folder
    if not os.path.isdir(input_folder):
        print(f"❌ Input folder '{input_folder}' does not exist or is not a directory.")
        return
    
    # Process the sketches
    process_sketches(input_folder, output_folder, not args.no_display, min_path_length)
    
    # If requested, try to open the first SVG with Illustrator
    if args.open_with_ai:
        vector_folder = os.path.join(output_folder, "vector")
        svg_files = [f for f in os.listdir(vector_folder) if f.endswith('.svg')]
        if svg_files:
            first_svg = os.path.join(vector_folder, svg_files[0])
            print(f"\n🚀 Attempting to open {first_svg} with Adobe Illustrator...")
            success = open_illustrator_with_file(first_svg)
            
            if not success:
                print("  📝 Please open the SVG files manually in Adobe Illustrator to continue.")

if __name__ == "__main__":
    print("🎨 Fashion Sketch to Tech Pack Pipeline 🎨")
    print("----------------------------------------")
    main()