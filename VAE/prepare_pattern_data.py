import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import glob
from tqdm import tqdm
import random
import argparse

def load_pattern_pieces(pattern_file):
    """
    Load pattern pieces from a JSON file
    
    Args:
        pattern_file: Path to pattern JSON file
    
    Returns:
        List of pattern pieces
    """
    try:
        with open(pattern_file, 'r') as f:
            data = json.load(f)
            
        # Check if the file contains pattern pieces
        if 'pattern_pieces' in data:
            return data['pattern_pieces']
        else:
            print(f"Warning: No pattern_pieces found in {pattern_file}")
            return []
    except Exception as e:
        print(f"Error loading {pattern_file}: {e}")
        return []

def normalize_pattern(points, target_size=(64, 64), padding=5):
    """
    Normalize pattern piece points to fit within target size
    
    Args:
        points: List of [x, y] coordinates
        target_size: Target size for normalization
        padding: Padding to add around the pattern
    
    Returns:
        Normalized points
    """
    if not points:
        return []
    
    # Convert to numpy array
    points_array = np.array(points)
    
    # Get min and max coordinates
    min_x, min_y = np.min(points_array, axis=0)
    max_x, max_y = np.max(points_array, axis=0)
    
    # Calculate scale factor to fit within target size with padding
    width = max_x - min_x
    height = max_y - min_y
    
    scale_x = (target_size[0] - 2 * padding) / width if width > 0 else 1
    scale_y = (target_size[1] - 2 * padding) / height if height > 0 else 1
    
    # Use the smaller scale to maintain aspect ratio
    scale = min(scale_x, scale_y)
    
    # Normalize points
    normalized_points = []
    for x, y in points:
        # Scale and center
        new_x = ((x - min_x) * scale) + padding
        new_y = ((y - min_y) * scale) + padding
        normalized_points.append([new_x, new_y])
    
    return normalized_points

def create_binary_image(points, image_size=(64, 64)):
    """
    Create a binary image from polygon points
    
    Args:
        points: List of [x, y] coordinates
        image_size: Size of the output image
    
    Returns:
        Binary image as numpy array
    """
    # Create a blank image
    img = np.zeros(image_size, dtype=np.uint8)
    
    if not points:
        return img
    
    # Convert points to integer array for OpenCV
    points_array = np.array(points, dtype=np.int32)
    
    # Draw the polygon
    cv2.fillPoly(img, [points_array], 255)
    
    return img

def augment_pattern(points, image_size=(64, 64)):
    """
    Apply random augmentations to pattern pieces
    
    Args:
        points: List of [x, y] coordinates
        image_size: Size of the output image
    
    Returns:
        List of augmented point sets
    """
    augmented_patterns = []
    
    # Original pattern
    augmented_patterns.append(points)
    
    # Rotation augmentations
    for angle in [90, 180, 270]:
        # Convert to numpy array
        points_array = np.array(points)
        
        # Get center of the pattern
        center_x = image_size[0] / 2
        center_y = image_size[1] / 2
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        
        # Apply rotation
        rotated_points = []
        for x, y in points_array:
            # Apply rotation matrix
            new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            rotated_points.append([new_x, new_y])
        
        augmented_patterns.append(rotated_points)
    
    # Flip augmentations
    points_array = np.array(points)
    
    # Horizontal flip
    flipped_h = [[image_size[0] - x, y] for x, y in points_array]
    augmented_patterns.append(flipped_h)
    
    # Vertical flip
    flipped_v = [[x, image_size[1] - y] for x, y in points_array]
    augmented_patterns.append(flipped_v)
    
    return augmented_patterns

def prepare_dataset(pattern_files, output_dir, image_size=(128, 128), augment=True):
    """
    Prepare dataset for VAE training
    
    Args:
        pattern_files: List of pattern JSON files
        output_dir: Directory to save processed data
        image_size: Size of the output images
        augment: Whether to apply data augmentation
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    all_pieces = []
    all_images = []
    
    for file_idx, pattern_file in enumerate(tqdm(pattern_files, desc="Processing pattern files")):
        # Load pattern pieces
        pieces = load_pattern_pieces(pattern_file)
        
        for piece_idx, piece in enumerate(pieces):
            if 'points' not in piece or not piece['points']:
                continue
            
            # Normalize points
            normalized_points = normalize_pattern(piece['points'], image_size)
            
            if augment:
                # Apply augmentations
                augmented_patterns = augment_pattern(normalized_points, image_size)
            else:
                augmented_patterns = [normalized_points]
            
            # Process each augmented pattern
            for aug_idx, aug_points in enumerate(augmented_patterns):
                # Create binary image
                binary_image = create_binary_image(aug_points, image_size)
                
                # Save image
                image_filename = f"piece_{file_idx}_{piece_idx}_{aug_idx}.png"
                cv2.imwrite(os.path.join(output_dir, 'images', image_filename), binary_image)
                
                # Add to dataset
                all_images.append(binary_image)
                
                # Save metadata
                piece_data = {
                    'id': f"piece_{file_idx}_{piece_idx}_{aug_idx}",
                    'original_id': piece.get('id', f"piece_{piece_idx}"),
                    'points': aug_points,
                    'image_file': image_filename,
                    'is_augmented': aug_idx > 0
                }
                all_pieces.append(piece_data)
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump({'pieces': all_pieces}, f, indent=2)
    
    # Save as numpy array for training
    all_images = np.array(all_images) / 255.0  # Normalize to [0, 1]
    all_images = all_images[..., np.newaxis]  # Add channel dimension
    np.save(os.path.join(output_dir, 'pattern_images.npy'), all_images)
    
    print(f"Processed {len(all_pieces)} pattern pieces ({len(all_images)} images)")
    print(f"Data saved to {output_dir}")
    
    # Visualize some examples
    visualize_samples(all_images, os.path.join(output_dir, 'sample_patterns.png'))

def visualize_samples(images, output_file, n_samples=10):
    """
    Visualize sample images
    
    Args:
        images: Array of images
        output_file: Output file path
        n_samples: Number of samples to visualize
    """
    n_samples = min(n_samples, len(images))
    indices = random.sample(range(len(images)), n_samples)
    
    plt.figure(figsize=(n_samples * 2, 2))
    for i, idx in enumerate(indices):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def find_pattern_files(data_dir, pattern="*.json"):
    """
    Find all pattern files in a directory
    
    Args:
        data_dir: Directory to search
        pattern: File pattern to match
    
    Returns:
        List of pattern file paths
    """
    return glob.glob(os.path.join(data_dir, pattern))

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare pattern data for VAE training')
    parser.add_argument('--input-dir', type=str, default='pattern_data',
                        help='Directory containing pattern JSON files')
    parser.add_argument('--output-dir', type=str, default='processed_patterns',
                        help='Directory to save processed data')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Size of the output images')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Find all pattern files
    pattern_files = find_pattern_files(args.input_dir)
    
    if not pattern_files:
        print(f"No pattern files found in {args.input_dir}")
        exit(1)
    
    print(f"Found {len(pattern_files)} pattern files")
    
    # Prepare dataset
    prepare_dataset(
        pattern_files=pattern_files,
        output_dir=args.output_dir,
        image_size=(args.image_size, args.image_size),
        augment=not args.no_augment
    )
