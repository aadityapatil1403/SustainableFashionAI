import os
import numpy as np
import cv2
import argparse
import json
import svgwrite
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A2
from reportlab.lib import colors
from reportlab.lib.units import mm
import time
import math
import random
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, box
from shapely.affinity import translate, rotate
from shapely import ops
import warnings

# Suppress shapely warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants for scrap analysis
MAX_RECTANGLE_COUNT = 100
LARGE_MASK_THRESHOLD = 1000

class ScrapAnalyzer:
    """
    Analyze fabric scraps to identify reusable areas.
    """
    def __init__(self, fabric_width, fabric_height, min_area=1000):
        self.fabric_width = int(fabric_width)
        self.fabric_height = int(fabric_height)
        self.min_area = min_area
        
    def analyze(self, pattern_pieces):
        """Analyze scrap in a pattern layout focusing on usable fabric metrics."""
        # Skip if dimensions are too small
        if self.fabric_width <= 10 or self.fabric_height <= 10:
            return self._create_empty_analysis()
                
        # Create an empty fabric mask (1 = empty, 0 = occupied)
        empty_mask = self._build_empty_space_mask(pattern_pieces)
        
        # Skip if empty mask is empty
        if np.sum(empty_mask) == 0:
            return self._create_empty_analysis()
        
        # Calculate total scrap area
        total_scrap_area = np.sum(empty_mask)
        
        # Find the smallest piece's area for threshold calculation
        if pattern_pieces:
            smallest_piece_area = min(piece.get('area', 5000) for piece in pattern_pieces)
            # Lower threshold to ensure rectangles are found
            min_usable_area = max(smallest_piece_area * 1.5, 2000)
        else:
            min_usable_area = 2000
                
        # Find multiple usable rectangles - limit to 10
        usable_rectangles = self._find_multiple_rectangles(empty_mask, min_usable_area, max_count=10)
        
        # Calculate total area of usable rectangles
        total_usable_area = sum(rect['area'] for rect in usable_rectangles)
        
        # Other calculations
        if usable_rectangles:
            max_rect_area = usable_rectangles[0]['area']
            max_rect_coords = usable_rectangles[0]['coords']
        else:
            max_rect_area = 0
            max_rect_coords = (0, 0, 0, 0)
                
        scrap_quality_index = (total_usable_area / total_scrap_area) * 100 if total_scrap_area > 0 else 0
        
        num_labels, _ = cv2.connectedComponents(empty_mask.astype(np.uint8), connectivity=8)
        scrap_regions = max(0, num_labels - 1)
        
        fabric_area = self.fabric_width * self.fabric_height
        total_usable_fabric = fabric_area - (total_scrap_area - total_usable_area)
        usable_fabric_ratio = total_usable_fabric / fabric_area if fabric_area > 0 else 0
        
        return {
            'total_scrap_area': total_scrap_area,
            'total_usable_area': total_usable_area,
            'largest_rectangle_area': max_rect_area,
            'largest_rectangle_coords': max_rect_coords,
            'usable_rectangles': usable_rectangles,
            'scrap_quality_index': scrap_quality_index,
            'scrap_regions': scrap_regions,
            'total_usable_fabric': total_usable_fabric,
            'usable_fabric_ratio': usable_fabric_ratio
        }
    
    def _create_empty_analysis(self):
        """Create an empty analysis result for edge cases."""
        return {
            'total_scrap_area': 0,
            'total_usable_area': 0,
            'largest_rectangle_area': 0,
            'largest_rectangle_coords': (0, 0, 0, 0),
            'usable_rectangles': [],
            'scrap_quality_index': 0,
            'scrap_regions': 0,
            'total_usable_fabric': 0,
            'usable_fabric_ratio': 0
        }
        
    def _build_empty_space_mask(self, pattern_pieces):
        """Build a binary mask of empty space in the layout."""
        # Create empty mask (all fabric is initially empty)
        mask = np.ones((self.fabric_height, self.fabric_width), dtype=np.uint8)
        
        # Fill in areas occupied by pattern pieces
        for piece in pattern_pieces:
            if 'new_x' not in piece or 'new_y' not in piece:
                continue
                
            try:
                # Get positioned polygon with rotation and translation
                positioned_poly = get_positioned_polygon(piece)
                
                # Convert polygon to integer coordinate points for OpenCV
                if positioned_poly.geom_type == 'Polygon':
                    points = np.array([list(map(int, p)) for p in positioned_poly.exterior.coords[:-1]], dtype=np.int32)
                    
                    # Check if all points are within bounds to avoid index errors
                    in_bounds = (
                        np.all(points[:, 0] >= 0) and 
                        np.all(points[:, 0] < self.fabric_width) and
                        np.all(points[:, 1] >= 0) and 
                        np.all(points[:, 1] < self.fabric_height)
                    )
                    
                    if in_bounds:
                        # Fill polygon in the mask (0 = occupied)
                        cv2.fillPoly(mask, [points], 0)
                    else:
                        # Try to clip the polygon to fit within bounds
                        clipped_points = []
                        for px, py in points:
                            clipped_points.append([
                                max(0, min(px, self.fabric_width - 1)),
                                max(0, min(py, self.fabric_height - 1))
                            ])
                        clipped_array = np.array(clipped_points, dtype=np.int32)
                        cv2.fillPoly(mask, [clipped_array], 0)
            except Exception as e:
                print(f"Error adding piece {piece.get('name', 'unknown')} to mask: {e}")
        
        return mask
    
    def _find_largest_rectangle(self, empty_mask):
        """Find the largest rectangle in the empty space with robust bounds checking."""
        height, width = empty_mask.shape
        
        # Early exit for empty masks
        if np.sum(empty_mask) == 0:
            return 0, (0, 0, 0, 0)
            
        # Downsample large masks for performance
        original_shape = None
        if height > LARGE_MASK_THRESHOLD or width > LARGE_MASK_THRESHOLD:
            scale_factor = LARGE_MASK_THRESHOLD / max(height, width)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            original_shape = (height, width)
            empty_mask = cv2.resize(empty_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            height, width = new_height, new_width
        
        # Use vectorized operations for histogram calculation
        max_area = 0
        max_rect = (0, 0, 0, 0)  # (x, y, w, h)
        
        # Initialize heights
        heights = np.zeros(width, dtype=np.int32)
        
        for row in range(height):
            # Update heights vector using vectorized operations
            heights = (heights + 1) * empty_mask[row, :]
            
            # Find largest rectangle in this histogram
            stack = []
            i = 0
            
            while i < width:
                if not stack or heights[stack[-1]] <= heights[i]:
                    stack.append(i)
                    i += 1
                else:
                    top = stack.pop()
                    
                    if not stack:
                        area = heights[top] * i
                    else:
                        area = heights[top] * (i - stack[-1] - 1)
                    
                    if area > max_area:
                        max_area = area
                        if not stack:
                            max_rect = (0, row - heights[top] + 1, i, heights[top])
                        else:
                            max_rect = (stack[-1] + 1, row - heights[top] + 1, i - stack[-1] - 1, heights[top])
            
            while stack:
                top = stack.pop()
                
                if not stack:
                    area = heights[top] * i
                else:
                    area = heights[top] * (i - stack[-1] - 1)
                
                if area > max_area:
                    max_area = area
                    if not stack:
                        max_rect = (0, row - heights[top] + 1, i, heights[top])
                    else:
                        max_rect = (stack[-1] + 1, row - heights[top] + 1, i - stack[-1] - 1, heights[top])
        
        # Scale rect back to original dimensions if needed
        if original_shape:
            scale_h = original_shape[0] / height
            scale_w = original_shape[1] / width
            x, y, w, h = max_rect
            scaled_x = int(x * scale_w)
            scaled_y = int(y * scale_h)
            scaled_w = int(w * scale_w)
            scaled_h = int(h * scale_h)
            
            if (scaled_x >= 0 and scaled_y >= 0 and 
                scaled_w > 0 and scaled_h > 0 and
                scaled_x + scaled_w <= original_shape[1] and 
                scaled_y + scaled_h <= original_shape[0]):
                max_rect = (scaled_x, scaled_y, scaled_w, scaled_h)
                max_area = int(max_area * scale_h * scale_w)
            else:
                max_rect = (0, 0, 0, 0)
                max_area = 0
        
        # Final bounds check
        if max_rect[0] < 0 or max_rect[1] < 0 or max_rect[2] <= 0 or max_rect[3] <= 0:
            return 0, (0, 0, 0, 0)
        
        return max_area, max_rect
    
    def _find_multiple_rectangles(self, empty_mask, min_area=1000, max_count=10):
        """Find multiple large rectangles in the empty space with optimized approach."""
        # Quick check for empty mask
        if np.sum(empty_mask) == 0:
            return []
            
        # Get mask dimensions for bounds checking
        height, width = empty_mask.shape
        
        # Make sure min_area is substantial
        min_area = max(min_area, 2000)
        
        # Make a copy of the original mask
        original_mask = empty_mask.copy()
        rectangles = []
        
        # Find rectangles iteratively - try more positions
        for i in range(max_count * 5):
            # Create a fresh working copy for this iteration
            mask = original_mask.copy()
            
            # Mask out already found rectangles
            for rect in rectangles:
                x, y, w, h = rect['coords']
                x_start = max(0, x)
                y_start = max(0, y)
                x_end = min(width, x + w)
                y_end = min(height, y + h)
                
                if x_start < x_end and y_start < y_end:
                    mask[y_start:y_end, x_start:x_end] = 0
            
            # Find largest rectangle in current mask
            rect_area, rect_coords = self._find_largest_rectangle(mask)
            x, y, w, h = rect_coords
            
            # Skip if rectangle is too small or invalid
            if (rect_area < min_area or w < 30 or h < 30 or 
                x < 0 or y < 0 or 
                x + w > width or y + h > height):
                continue
            
            # Add valid rectangle
            rectangles.append({
                'area': rect_area,
                'coords': rect_coords,
                'value_index': i
            })
            
            # Exit when we have enough rectangles
            if len(rectangles) >= max_count:
                break
        
        # Sort rectangles by area (largest first)
        rectangles.sort(key=lambda r: r['area'], reverse=True)
        
        return rectangles[:max_count]

def calculate_original_fabric_area(pattern_pieces):
    """
    Calculate the area of the original layout.
    """
    if not pattern_pieces:
        return 0, 0
        
    # Find bounds of original layout
    min_x = min(min(x for x, _ in piece['points']) for piece in pattern_pieces)
    min_y = min(min(y for _, y in piece['points']) for piece in pattern_pieces)
    max_x = max(max(x for x, _ in piece['points']) for piece in pattern_pieces)
    max_y = max(max(y for _, y in piece['points']) for piece in pattern_pieces)
    
    width = max_x - min_x
    height = max_y - min_y
    
    return width, height

def calculate_smallest_enclosing_rectangle(pattern_pieces):
    """
    Calculate the smallest rectangle that encloses all placed pattern pieces.
    """
    if not pattern_pieces:
        return 0, 0, 0, 0
        
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for piece in pattern_pieces:
        if 'new_x' not in piece or 'new_y' not in piece:
            continue
            
        # Get positioned polygon
        positioned_poly = get_positioned_polygon(piece)
        
        # Update bounds
        piece_min_x, piece_min_y, piece_max_x, piece_max_y = positioned_poly.bounds
        min_x = min(min_x, piece_min_x)
        min_y = min(min_y, piece_min_y)
        max_x = max(max_x, piece_max_x)
        max_y = max(max_y, piece_max_y)
    
    width = max_x - min_x
    height = max_y - min_y
    
    return width, height, min_x, min_y

def extract_pattern_pieces(pdf_path):
    """Extract pattern pieces from a PDF file."""
    print(f"Converting PDF to image: {pdf_path}")
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=150)
    
    # Process the first page
    img = np.array(images[0])
    # Convert RGB to BGR (for OpenCV)
    img = img[:, :, ::-1].copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pattern_pieces = []
    for i, contour in enumerate(contours):
        # Filter small contours (noise)
        if cv2.contourArea(contour) < 5000:
            continue
        
        # Simplify contour (reduce points)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to simple polygon
        points = [(int(point[0][0]), int(point[0][1])) for point in approx]
        
        # Create Shapely polygon for NFP calculations
        polygon = Polygon(points)
        
        # Calculate bounding box for initial placement
        x, y, w, h = cv2.boundingRect(approx)
        
        # Create pattern piece object
        piece = {
            'name': f'piece_{i}',
            'points': points,
            'polygon': polygon,
            'width': w,
            'height': h,
            'x': x,  # original x position
            'y': y,  # original y position
            'area': polygon.area,  # Using shapely area calculation
            'rotated': False,  # Track rotation state
            'rotation_angle': 0  # Track rotation angle in degrees
        }
        
        pattern_pieces.append(piece)
    
    print(f"Extracted {len(pattern_pieces)} pattern pieces")
    return pattern_pieces, img.shape[1], img.shape[0]

def compute_minkowski_difference(stationary_poly, moving_poly):
    """
    Compute the Minkowski difference between two polygons.
    This is equivalent to calculating the No-Fit Polygon.
    """
    # Ensure we're working with valid polygons
    if not stationary_poly.is_valid:
        stationary_poly = stationary_poly.buffer(0)
    if not moving_poly.is_valid:
        moving_poly = moving_poly.buffer(0)
    
    # Create a reflected (mirrored) version of the moving polygon
    moving_poly_reflected = Polygon([(-x, -y) for x, y in moving_poly.exterior.coords])
    
    # Calculate the Minkowski sum of stationary and reflected moving poly
    # This is equivalent to the Minkowski difference
    nfp = stationary_poly.convex_hull.union(moving_poly_reflected.convex_hull)
    
    # Handle potential topology errors
    if not nfp.is_valid:
        nfp = nfp.buffer(0)
    
    return nfp

def compute_inner_fit_polygon(container_poly, moving_poly):
    """
    Compute the Inner Fit Polygon (IFP) - the positions where the moving polygon
    can be placed inside the container polygon without overlapping the edges.
    """
    # Shrink the container by the moving polygon dimensions (simplified)
    # This is a rough approximation for demonstration purposes
    ifp = container_poly.buffer(-max(moving_poly.bounds[2] - moving_poly.bounds[0], 
                                    moving_poly.bounds[3] - moving_poly.bounds[1]) / 2)
    
    # Return the IFP, or an empty polygon if the buffer operation fails
    if ifp.is_empty:
        return Polygon()
    return ifp

def generate_rotation_angles(min_angle=0, max_angle=360, step=15, fine_tuning=True):
    """
    Generate a list of rotation angles to try.
    Includes standard angles plus some finer increments for better optimization.
    
    Args:
        min_angle: Minimum angle in degrees
        max_angle: Maximum angle in degrees
        step: Step size for standard angles
        fine_tuning: Whether to include finer increments around 90-degree angles
        
    Returns:
        List of angles to try
    """
    # Generate standard angles at regular intervals
    standard_angles = list(range(min_angle, max_angle, step))
    
    if fine_tuning:
        # Add fine-tuned angles around key orientations (0, 90, 180, 270 degrees)
        fine_angles = []
        for base in [0, 90, 180, 270]:
            for fine_step in [-10, -5, 5, 10]:
                fine_angle = (base + fine_step) % 360
                if fine_angle not in standard_angles:
                    fine_angles.append(fine_angle)
        
        # Combine and sort all angles
        all_angles = sorted(standard_angles + fine_angles)
    else:
        all_angles = standard_angles
    
    return all_angles

def get_nfp_placement_positions(placed_pieces, current_piece, fabric_width, fabric_height, grid_step=10, use_arbitrary_rotation=False, rotation_step=15):
    """
    Generate potential placement positions for the current piece using NFPs.
    """
    # Generate rotation angles based on the arbitrary rotation setting
    if use_arbitrary_rotation:
        rotation_angles = generate_rotation_angles(step=rotation_step)
    else:
        # Use only 90-degree rotations if arbitrary rotation is not enabled
        rotation_angles = [0, 90, 180, 270]
    
    # If no pieces have been placed yet, create a grid of starting positions
    if not placed_pieces:
        positions = []
        for x in range(0, fabric_width, grid_step):
            for y in range(0, int(fabric_height / 4), grid_step):  # Start at the top for better packing
                # Add positions with various rotation angles
                for angle in rotation_angles:
                    # Check if the piece would fit at this position with this rotation
                    current_poly = current_piece['polygon']
                    if angle != 0:
                        centroid = current_poly.centroid
                        current_poly = rotate(current_poly, angle, origin=centroid)
                    
                    # Calculate translation
                    dx = x - current_piece['x']
                    dy = y - current_piece['y']
                    moved_poly = translate(current_poly, dx, dy)
                    
                    # Get bounds
                    minx, miny, maxx, maxy = moved_poly.bounds
                    
                    # Only add if it fits within the fabric
                    if (minx >= 0 and miny >= 0 and 
                        maxx <= fabric_width and maxy <= fabric_height):
                        positions.append((x, y, angle))
        return positions
    
    # Create a container polygon representing the fabric
    fabric_poly = Polygon([(0, 0), (fabric_width, 0), 
                           (fabric_width, fabric_height), (0, fabric_height)])
    
    # Get the current piece's polygon
    current_poly = current_piece['polygon']
    
    # List to collect valid positions from NFP calculations
    nfp_positions = []
    
    # For each placed piece, compute the NFP
    for placed_piece in placed_pieces:
        # Get the placed polygon in its final position and rotation
        placed_poly = get_positioned_polygon(placed_piece)
        
        # Compute NFP for current position
        nfp = compute_minkowski_difference(placed_poly, current_poly)
        
        # Extract the NFP boundary points for potential placement positions
        # We use the exterior coordinates of the NFP as placement positions
        try:
            if isinstance(nfp, MultiPolygon):
                for poly in nfp.geoms:
                    for coord in poly.exterior.coords:
                        # For each potential position, check if the piece would fit within bounds
                        for angle in rotation_angles:
                            # Apply rotation
                            test_poly = current_poly
                            if angle != 0:
                                centroid = test_poly.centroid
                                test_poly = rotate(test_poly, angle, origin=centroid)
                            
                            # Apply translation
                            dx = coord[0] - current_piece['x']
                            dy = coord[1] - current_piece['y']
                            moved_poly = translate(test_poly, dx, dy)
                            
                            # Check bounds
                            minx, miny, maxx, maxy = moved_poly.bounds
                            if (minx >= 0 and miny >= 0 and 
                                maxx <= fabric_width and maxy <= fabric_height):
                                nfp_positions.append((coord[0], coord[1], angle))
            else:
                for coord in nfp.exterior.coords:
                    # For each potential position, check if the piece would fit within bounds
                    for angle in rotation_angles:
                        # Apply rotation
                        test_poly = current_poly
                        if angle != 0:
                            centroid = test_poly.centroid
                            test_poly = rotate(test_poly, angle, origin=centroid)
                        
                        # Apply translation
                        dx = coord[0] - current_piece['x']
                        dy = coord[1] - current_piece['y']
                        moved_poly = translate(test_poly, dx, dy)
                        
                        # Check bounds
                        minx, miny, maxx, maxy = moved_poly.bounds
                        if (minx >= 0 and miny >= 0 and 
                            maxx <= fabric_width and maxy <= fabric_height):
                            nfp_positions.append((coord[0], coord[1], angle))
        except Exception as e:
            print(f"NFP extraction error: {e}")
            # Fall back to a grid-based approach on error
            continue
    
    # Add positions from the inner fit polygon (positions valid inside the fabric)
    ifp = compute_inner_fit_polygon(fabric_poly, current_poly)
    
    # If we couldn't generate any valid positions, fall back to a grid
    if len(nfp_positions) < 10:
        print(f"Few NFP positions found ({len(nfp_positions)}), adding grid positions")
        for x in range(0, fabric_width, grid_step * 2):
            for y in range(0, fabric_height, grid_step * 2):
                # Add positions with various rotation angles
                for angle in rotation_angles:
                    # Check if the piece would fit at this position with this rotation
                    test_poly = current_piece['polygon']
                    if angle != 0:
                        centroid = test_poly.centroid
                        test_poly = rotate(test_poly, angle, origin=centroid)
                    
                    # Apply translation
                    dx = x - current_piece['x']
                    dy = y - current_piece['y']
                    moved_poly = translate(test_poly, dx, dy)
                    
                    # Check bounds
                    minx, miny, maxx, maxy = moved_poly.bounds
                    if (minx >= 0 and miny >= 0 and 
                        maxx <= fabric_width and maxy <= fabric_height):
                        nfp_positions.append((x, y, angle))
    
    # Remove duplicates and limit the number of positions to avoid excessive computation
    # Convert positions to hashable tuples for deduplication
    unique_positions = list(set((round(x), round(y), angle) for x, y, angle in nfp_positions))
    
    # Limit the number of positions to evaluate
    max_positions = 500  # Increased from 200 to allow for more rotation angles
    if len(unique_positions) > max_positions:
        # If we have too many positions, use a combination of random sampling and key angles
        # This ensures we don't just evaluate positions with the same angle
        
        # First, separate positions by angle
        positions_by_angle = {}
        for pos in unique_positions:
            angle = pos[2]
            if angle not in positions_by_angle:
                positions_by_angle[angle] = []
            positions_by_angle[angle].append(pos)
        
        # Sample positions from each angle
        sampled_positions = []
        for angle, positions in positions_by_angle.items():
            # Take more samples from 0 and 90 degree rotations (these are often better)
            sample_count = min(len(positions), 50 if angle in [0, 90, 180, 270] else 20)
            sampled_positions.extend(random.sample(positions, sample_count))
        
        # If we still have too many, randomly sample
        if len(sampled_positions) > max_positions:
            unique_positions = random.sample(sampled_positions, max_positions)
        else:
            unique_positions = sampled_positions
    
    return unique_positions

def get_positioned_polygon(piece):
    """
    Get the Shapely polygon for a piece in its final position and rotation.
    """
    # Get the original polygon
    polygon = piece['polygon']
    
    # Calculate position difference
    if 'new_x' in piece and 'new_y' in piece:
        dx = piece['new_x'] - piece['x']
        dy = piece['new_y'] - piece['y']
    else:
        dx = 0
        dy = 0
    
    # Apply rotation if needed
    if piece.get('rotated', False) and 'rotation_angle' in piece:
        # Get centroid for rotation
        centroid = polygon.centroid
        
        # Rotate around centroid
        rotated = rotate(polygon, piece['rotation_angle'], origin=centroid)
        
        # Translate to final position
        positioned = translate(rotated, dx, dy)
    else:
        # Just translate if no rotation
        positioned = translate(polygon, dx, dy)
    
    return positioned

def detect_collision(piece, placed_pieces, x, y, rotation_angle=0):
    """
    Detect if a piece would collide with already placed pieces or the fabric boundaries.
    """
    # Get the piece's polygon
    current_poly = piece['polygon']
    
    # Apply rotation if needed
    if rotation_angle != 0:
        # Rotate around centroid
        centroid = current_poly.centroid
        current_poly = rotate(current_poly, rotation_angle, origin=centroid)
    
    # Translate to proposed position
    dx = x - piece['x']
    dy = y - piece['y']
    moved_poly = translate(current_poly, dx, dy)
    
    # Check for collisions with each placed piece
    for placed_piece in placed_pieces:
        placed_poly = get_positioned_polygon(placed_piece)
        
        # Check if polygons intersect
        if moved_poly.intersects(placed_poly):
            return True  # Collision detected
    
    return False  # No collision

def calculate_bounding_box(polygon):
    """Calculate the bounding box of a rotated polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    return (maxx - minx), (maxy - miny), minx, miny

class NFPPacker:
    """
    Enhanced pattern piece packing algorithm using No-Fit Polygons.
    """
    
    def __init__(self, fabric_width, fabric_height, allow_rotation=True, arbitrary_rotation=False, margin=0, rotation_step=15):
        self.fabric_width = fabric_width
        self.fabric_height = fabric_height
        self.allow_rotation = allow_rotation
        self.arbitrary_rotation = arbitrary_rotation
        self.margin = margin
        self.placed_pieces = []
        self.grid_step = 20  # Step size for grid search
        self.rotation_step = rotation_step
        
        # Print rotation settings
        if allow_rotation:
            if arbitrary_rotation:
                print(f"Using arbitrary rotation with step size {rotation_step}°")
            else:
                print("Using standard 90° rotations (0°, 90°, 180°, 270°)")
        else:
            print("Rotation disabled")
        
    def fit(self, blocks):
        """Fit blocks into the bin using NFP-based placement."""
        if not blocks:
            return
        
        # Apply margins to block dimensions
        for block in blocks:
            block['margin_width'] = block['width'] + 2 * self.margin
            block['margin_height'] = block['height'] + 2 * self.margin
            
            # Create a buffered polygon for margin
            if self.margin > 0:
                block['margin_polygon'] = block['polygon'].buffer(self.margin)
            else:
                block['margin_polygon'] = block['polygon']
        
        # Sort blocks by area (largest first)
        blocks.sort(key=lambda b: b['area'], reverse=True)
        
        for i, block in enumerate(blocks):
            print(f"Placing piece {i+1}/{len(blocks)}: {block['name']}")
            success = self.place_piece(block)
            
            if not success:
                print(f"Warning: Could not place piece {block['name']}")
    
    def place_piece(self, piece):
        """
        Find the best position for a piece using NFP and place it.
        """
        # Get potential placement positions
        positions = get_nfp_placement_positions(
            self.placed_pieces, piece, 
            self.fabric_width, self.fabric_height, 
            self.grid_step,
            self.arbitrary_rotation,
            self.rotation_step
        )
        
        # If no positions found, return failure
        if not positions:
            return False
        
        best_position = None
        best_score = float('-inf')
        
        # Try all positions and find the best one
        for pos in positions:
            x, y, rotation_angle = pos
            
            # Skip if position would cause the piece to go out of bounds
            # We need to calculate the bounding box differently for arbitrary rotations
            current_poly = piece['polygon']
            
            # Apply rotation
            if rotation_angle != 0:
                centroid = current_poly.centroid
                current_poly = rotate(current_poly, rotation_angle, origin=centroid)
            
            # Get the bounds of the rotated polygon
            # Apply translation to get the actual position
            dx = x - piece['x']
            dy = y - piece['y']
            moved_poly = translate(current_poly, dx, dy)
            
            # Get the actual bounds of the positioned polygon
            minx, miny, maxx, maxy = moved_poly.bounds
            
            # Check if the piece would go out of bounds
            if (minx < 0 or miny < 0 or 
                maxx > self.fabric_width or 
                maxy > self.fabric_height):
                continue
            
            # Skip if collision detected
            if detect_collision(piece, self.placed_pieces, x, y, rotation_angle):
                continue
            
            # Calculate score for this position
            score = self.evaluate_placement(piece, x, y, rotation_angle)
            
            # Update best position if this one is better
            if score > best_score:
                best_score = score
                best_position = (x, y, rotation_angle)
        
        # If a valid position was found, place the piece
        if best_position:
            x, y, rotation_angle = best_position
            
            # Update piece with new position and rotation
            piece['new_x'] = x
            piece['new_y'] = y
            piece['rotated'] = rotation_angle != 0
            piece['rotation_angle'] = rotation_angle
            
            # Add to placed pieces
            self.placed_pieces.append(piece)
            return True
        
        # If we couldn't place the piece normally, try the emergency placement
        return self.emergency_placement(piece)
    
    def evaluate_placement(self, piece, x, y, rotation_angle):
        """
        Evaluate how good a placement is (higher is better).
        This function prioritizes placements that:
        1. Are close to other pieces (for tighter packing)
        2. Are higher on the fabric (toward y=0)
        3. Are toward the left of the fabric (toward x=0)
        4. Minimize total fabric area
        """
        # Initialize score components
        adjacency_score = 0
        height_score = 0
        left_alignment_score = 0
        fabric_utilization_score = 0
        
        # 1. Calculate adjacency to other pieces
        if self.placed_pieces:
            min_distance = float('inf')
            current_poly = piece['polygon']
            
            # Apply rotation if needed
            if rotation_angle != 0:
                centroid = current_poly.centroid
                current_poly = rotate(current_poly, rotation_angle, origin=centroid)
            
            # Translate to proposed position
            moved_poly = translate(current_poly, x - piece['x'], y - piece['y'])
            
            for placed_piece in self.placed_pieces:
                placed_poly = get_positioned_polygon(placed_piece)
                
                # Minimum distance between polygons
                try:
                    distance = moved_poly.distance(placed_poly)
                    min_distance = min(min_distance, distance)
                    
                    # Bonus for touching pieces (adjacency)
                    if distance < 0.1:  # Almost touching
                        adjacency_score += 100
                except Exception:
                    # Fallback if Shapely has issues
                    continue
            
            # Convert distance to score (closer is better)
            if min_distance != float('inf'):
                adjacency_score += 50 / (min_distance + 1)
        
        # 2. Prefer placements higher on the fabric (smaller y)
        height_score = 100 * (1 - y / self.fabric_height)
        
        # 3. Prefer placements toward the left (smaller x)
        left_alignment_score = 50 * (1 - x / self.fabric_width)
        
        # 4. Calculate how this placement affects the total fabric area
        current_poly = piece['polygon']
        if rotation_angle != 0:
            centroid = current_poly.centroid
            current_poly = rotate(current_poly, rotation_angle, origin=centroid)
        moved_poly = translate(current_poly, x - piece['x'], y - piece['y'])
        
        # Calculate the maximum y coordinate after placement
        _, _, _, max_y = moved_poly.bounds
        
        # Find current maximum height
        current_max_height = 0
        for placed_piece in self.placed_pieces:
            placed_poly = get_positioned_polygon(placed_piece)
            _, _, _, placed_max_y = placed_poly.bounds
            current_max_height = max(current_max_height, placed_max_y)
        
        # Calculate how much this placement increases the fabric height
        height_increase = max(0, max_y - current_max_height)
        
        # Give bonus for placements that don't increase fabric height
        if height_increase == 0:
            fabric_utilization_score = 150
        else:
            # Penalize placements that increase fabric height
            fabric_utilization_score = 100 / (height_increase + 1)
        
        # Combine scores
        total_score = adjacency_score + height_score + left_alignment_score + fabric_utilization_score
        
        # Add bonus for rotation if it's beneficial
        if self.allow_rotation and rotation_angle != 0:
            # Bonus for rotating tall/wide pieces to fit better
            if (piece['width'] > piece['height'] and rotation_angle in [90, 270, 45, 135, 225, 315]) or \
               (piece['height'] > piece['width'] and rotation_angle in [0, 180, 45, 135, 225, 315]):
                total_score += 30
            
            # Slightly prefer standard 90° rotations for manufacturing simplicity
            if rotation_angle in [0, 90, 180, 270]:
                total_score += 20
            
            # Small bonus for rotation angles that align well with the fabric edges
            if rotation_angle % 90 == 0:
                total_score += 10
        
        return total_score
    
    def emergency_placement(self, piece):
        """
        Last resort placement for pieces that couldn't be placed normally.
        Just puts the piece at the bottom of the fabric.
        """
        print(f"Using emergency placement for piece {piece['name']}")
        
        # Try to place at the bottom of the current layout
        max_y = 0
        for placed_piece in self.placed_pieces:
            placed_poly = get_positioned_polygon(placed_piece)
            _, _, _, max_y_poly = placed_poly.bounds
            max_y = max(max_y, max_y_poly)
        
        # Place the piece at the bottom with some spacing
        y = max_y + self.margin * 2
        
        # Try different x positions
        for x in range(0, self.fabric_width - int(piece['width']), 20):
            if not detect_collision(piece, self.placed_pieces, x, y):
                piece['new_x'] = x
                piece['new_y'] = y
                piece['rotated'] = False
                piece['rotation_angle'] = 0
                self.placed_pieces.append(piece)
                return True
        
        # If still not placed, try with rotation if allowed
        if self.allow_rotation:
            # Try key rotation angles for emergency placement
            rotation_angles = [90, 45, 135, 180, 270, 315, 225] if self.arbitrary_rotation else [90, 180, 270]
            
            for angle in rotation_angles:
                # Calculate bounding box for the rotated piece
                current_poly = piece['polygon']
                centroid = current_poly.centroid
                rotated_poly = rotate(current_poly, angle, origin=centroid)
                
                for x in range(0, self.fabric_width, 20):
                    # Check if the piece would fit at this position with this rotation
                    dx = x - piece['x']
                    moved_poly = translate(rotated_poly, dx, y - piece['y'])
                    
                    # Check bounds
                    minx, miny, maxx, maxy = moved_poly.bounds
                    if not (minx >= 0 and miny >= 0 and 
                            maxx <= self.fabric_width and maxy <= self.fabric_height):
                        continue
                    if not detect_collision(piece, self.placed_pieces, x, y, angle):
                        piece['new_x'] = x
                        piece['new_y'] = y
                        piece['rotated'] = True
                        piece['rotation_angle'] = angle
                        self.placed_pieces.append(piece)
                        return True
        
        # If all else fails, grow the fabric height and place at the bottom
        y = self.fabric_height
        self.fabric_height += piece['height'] + self.margin * 2
        
        piece['new_x'] = 0
        piece['new_y'] = y
        piece['rotated'] = False
        piece['rotation_angle'] = 0
        self.placed_pieces.append(piece)
        return True

def create_svg(pattern_pieces, fabric_width, fabric_height, output_path, margin=0):
    """Create SVG visualization of the pattern layout."""
    # Create SVG drawing
    dwg = svgwrite.Drawing(output_path, profile='tiny')
    dwg.viewbox(0, 0, fabric_width, fabric_height)
    
    # Add fabric background
    dwg.add(dwg.rect((0, 0), (fabric_width, fabric_height), fill='white', stroke='black', stroke_width=1))
    
    # Add margin grid if margin is specified
    if margin > 0:
        # Add a light grid to show margins
        grid_color = '#EEEEEE'
        for x in range(0, fabric_width, margin):
            dwg.add(dwg.line((x, 0), (x, fabric_height), stroke=grid_color, stroke_width=0.5))
        for y in range(0, fabric_height, margin):
            dwg.add(dwg.line((0, y), (fabric_width, y), stroke=grid_color, stroke_width=0.5))
    
    # Color palette for pieces
    colors = ['#FF5555', '#55FF55', '#5555FF', '#FFFF55', '#FF55FF', '#55FFFF', 
              '#AA5555', '#55AA55', '#5555AA', '#AAAA55', '#AA55AA', '#55AAAA']
    
    # Add pattern pieces
    for i, piece in enumerate(pattern_pieces):
        if 'new_x' not in piece or 'new_y' not in piece:
            continue
            
        color = colors[i % len(colors)]
        
        # Get positioned polygon
        poly = get_positioned_polygon(piece)
        
        # Extract points for SVG
        svg_points = [(coord[0], coord[1]) for coord in poly.exterior.coords]
        
        # Add polygon
        dwg.add(dwg.polygon(svg_points, fill=color, stroke='black', stroke_width=1))
        
        # Add label with piece name and rotation indicator
        rotation_indicator = f" (R{piece['rotation_angle']}°)" if piece.get('rotated', False) else ""
        
        # Calculate centroid for text placement
        centroid = poly.centroid
        dwg.add(dwg.text(piece['name'] + rotation_indicator, insert=(centroid.x, centroid.y), 
                         text_anchor='middle', font_size='12'))
    
    # Save drawing
    dwg.save()
    print(f"SVG saved to {output_path}")
    
    return output_path

def create_pdf(svg_path, original_width, original_height, fabric_width, fabric_height,
               pattern_pieces, original_pieces, original_analysis, optimized_analysis, stats, output_pdf, margin=0):
    """Create PDF with original and optimized layouts plus comparison statistics."""
    # Use A2 paper size for consistency with other PDFs
    page_width, page_height = A2
    c = canvas.Canvas(output_pdf, pagesize=A2)
    
    # Calculate margins and layout positioning
    margin_mm = 20 * mm
    available_height = page_height - 2 * margin_mm
    available_width = page_width - 2 * margin_mm
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_mm, page_height - margin_mm, "Zero Waste Pattern Piece Optimization")
    
    # Add statistics
    c.setFont("Helvetica", 10)
    y_pos = page_height - margin_mm - 20 * mm
    
    # Draw comparison statistics
    section = None
    in_section = False
    
    for line in stats:
        # Handle section headers
        if line.startswith("**=="):
            # End previous section if needed
            if in_section:
                c.restoreState()
                in_section = False
            
            # Start new section
            section = line.strip("*= ")
            c.saveState()
            in_section = True
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_mm, y_pos, line.replace("**", ""))
            y_pos -= 6 * mm
            c.setFont("Helvetica", 10)  # Reset font for content after header
            continue
        
        # Handle empty lines
        if not line:
            y_pos -= 3 * mm
            continue
        
        # Draw regular stat line
        c.drawString(margin_mm, y_pos, line)
        y_pos -= 4.5 * mm
    
    # End last section if still active
    if in_section:
        c.restoreState()
        in_section = False
    
    # Dividing line
    y_pos -= 5 * mm
    c.line(margin_mm, y_pos, page_width - margin_mm, y_pos)
    y_pos -= 10 * mm
    
    # Calculate max height needed for displays
    layout_height = max(original_height, fabric_height)
    
    # Scale factors to fit layouts side by side
    max_layout_width = (available_width - 20 * mm) / 2
    max_layout_height = available_height - (page_height - y_pos)
    
    scale_original = min(max_layout_width / original_width, max_layout_height / layout_height) if original_width > 0 and layout_height > 0 else 1
    scale_optimized = min(max_layout_width / fabric_width, max_layout_height / layout_height) if fabric_width > 0 and layout_height > 0 else 1
    
    # Use unified scale for visual consistency
    unified_scale = min(scale_original, scale_optimized)
    
    # Section headers - center both titles
    c.setFont("Helvetica-Bold", 14)
    left_center_x = margin_mm + max_layout_width/2
    right_center_x = margin_mm + max_layout_width + 20*mm + max_layout_width/2
    
    # Draw centered titles
    c.drawCentredString(left_center_x, y_pos, "Original Layout")
    c.drawCentredString(right_center_x, y_pos, "Optimized Layout")
    y_pos -= 8 * mm
    
    # Calculate starting positions for layouts
    orig_start_y = y_pos - unified_scale * layout_height
    optim_start_y = y_pos - unified_scale * layout_height
    
    # Draw original layout rectangle
    original_layout_x = margin_mm
    original_layout_y = orig_start_y
    c.rect(original_layout_x, original_layout_y, 
           unified_scale * original_width, unified_scale * layout_height, 
           stroke=1, fill=0)
    
    # Draw optimized layout rectangle
    optimized_layout_x = margin_mm + max_layout_width + 20*mm
    optimized_layout_y = optim_start_y
    c.rect(optimized_layout_x, optimized_layout_y, 
           unified_scale * fabric_width, unified_scale * layout_height, 
           stroke=1, fill=0)
    
    # Draw original layout
    c.saveState()
    c.translate(original_layout_x, original_layout_y)
    c.scale(unified_scale, unified_scale)
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0.97, 0.97, 0.97)  
    c.rect(0, 0, original_width, original_height, fill=1, stroke=1)
    
    # Draw original pieces
    for i, piece in enumerate(original_pieces):
        r, g, b = [(i % 3 == 0) * 0.8, (i % 3 == 1) * 0.8, (i % 3 == 2) * 0.8]
        c.setFillColorRGB(r, g, b, 0.8)
        c.setStrokeColorRGB(0, 0, 0)
        c.setDash(1, 0)
        
        if piece['polygon'].geom_type == 'Polygon':
            path = c.beginPath()
            coords = list(piece['polygon'].exterior.coords)
            path.moveTo(coords[0][0], coords[0][1])
            for point in coords[1:]:
                path.lineTo(point[0], point[1])
            path.close()
            c.drawPath(path, fill=1, stroke=1)
    
    # Draw the green rectangles
    if 'usable_rectangles' in original_analysis:
        c.setFillColorRGB(0, 0.8, 0, 0.4)
        c.setStrokeColorRGB(0, 0.7, 0, 0.9)
        c.setDash(5, 5)
        for rect_data in original_analysis['usable_rectangles'][:10]:
            rect = rect_data['coords']
            x, y, w, h = rect
            if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= original_width and y + h <= original_height):
                c.rect(x, y, w, h, fill=1, stroke=1)
    
    c.restoreState()
    
    # Draw optimized layout
    c.saveState()
    c.translate(optimized_layout_x, optimized_layout_y)
    c.scale(unified_scale, unified_scale)
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0.97, 0.97, 0.97)
    c.rect(0, 0, fabric_width, fabric_height, fill=1, stroke=1)
    
    # Draw optimized pieces
    for i, piece in enumerate(pattern_pieces):
        if 'new_x' not in piece or 'new_y' not in piece:
            continue
            
        r, g, b = [(i % 3 == 0) * 0.8, (i % 3 == 1) * 0.8, (i % 3 == 2) * 0.8]
        c.setFillColorRGB(r, g, b, 0.8)
        c.setStrokeColorRGB(0, 0, 0)
        c.setDash(1, 0)
        
        positioned_poly = get_positioned_polygon(piece)
        
        if positioned_poly.geom_type == 'Polygon':
            path = c.beginPath()
            coords = list(positioned_poly.exterior.coords)
            path.moveTo(coords[0][0], coords[0][1])
            for point in coords[1:]:
                path.lineTo(point[0], point[1])
            path.close()
            c.drawPath(path, fill=1, stroke=1)
        
        # Add rotation indicator
        if piece.get('rotated', False):
            c.setFont("Helvetica", 8)
            c.setFillColorRGB(0, 0, 0)
            centroid = positioned_poly.centroid
            c.drawCentredString(centroid.x, centroid.y, f"(R{piece['rotation_angle']}°)")
    
    # Draw green rectangles
    if 'usable_rectangles' in optimized_analysis:
        c.setFillColorRGB(0, 0.8, 0, 0.4)
        c.setStrokeColorRGB(0, 0.7, 0, 0.9)
        c.setDash(5, 5)
        for rect_data in optimized_analysis['usable_rectangles']:
            rect = rect_data['coords']
            x, y, w, h = rect
            if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= fabric_width and y + h <= fabric_height):
                c.rect(x, y, w, h, fill=1, stroke=1)
    
    c.restoreState()
    
    # Add footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(margin_mm, margin_mm/2, "Green highlighted rectangles represent reusable fabric pieces")
    
    # Save the PDF
    c.save()
    print(f"PDF saved to {output_pdf}")

def optimize_pattern_layout(input_pdf, output_prefix, fabric_width=None, allow_rotation=True, arbitrary_rotation=False, margin=0, rotation_step=15):
    """Main function to optimize pattern layout using NFP."""
    start_time = time.time()
    
    # Extract pattern pieces from PDF
    pattern_pieces, original_width, original_height = extract_pattern_pieces(input_pdf)
    
    # Create a copy of original pattern pieces for comparison
    original_pieces = []
    for piece in pattern_pieces:
        original_piece = piece.copy()
        original_piece['polygon'] = Polygon(original_piece['points'])
        original_piece['new_x'] = original_piece['x']
        original_piece['new_y'] = original_piece['y']
        original_pieces.append(original_piece)
    
    if not pattern_pieces:
        print("No pattern pieces were extracted.")
        return
    
    # Use a reasonable fabric width if not provided
    if fabric_width is None:
        fabric_width = int(original_width * 0.9)
    
    # Use a reasonable initial height
    initial_height = int(sum(piece['height'] for piece in pattern_pieces[:3]) * 1.5)
    
    print(f"Using fabric width: {fabric_width}")
    print(f"Initial fabric height: {initial_height}")
    
    # Create the NFP packer
    packer = NFPPacker(fabric_width, initial_height, allow_rotation=allow_rotation, 
                      arbitrary_rotation=arbitrary_rotation, margin=margin,
                      rotation_step=rotation_step)
    
    # Run NFP packing
    packer.fit(pattern_pieces)
    
    # Check how many pieces were placed successfully
    placed_pieces = sum(1 for piece in pattern_pieces if 'new_x' in piece and 'new_y' in piece)
    total_pieces = len(pattern_pieces)
    
    # Calculate the true bounding box of all placed pieces
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for piece in pattern_pieces:
        if 'new_x' not in piece or 'new_y' not in piece:
            continue
        
        poly = get_positioned_polygon(piece)
        piece_min_x, piece_min_y, piece_max_x, piece_max_y = poly.bounds
        
        min_x = min(min_x, piece_min_x)
        min_y = min(min_y, piece_min_y)
        max_x = max(max_x, piece_max_x)
        max_y = max(max_y, piece_max_y)
    
    # Calculate the actual used dimensions (not the container)
    actual_width = max_x - min_x + (margin * 2)
    actual_height = max_y - min_y + (margin * 2)
    
    # For container dimensions (needed for visualization)
    final_fabric_width = fabric_width
    final_fabric_height = max(packer.fabric_height, int(actual_height))
    
    # Calculate the actual used area in both layouts
    # For original layout, use the bounding box of original pieces
    orig_min_x = float('inf')
    orig_min_y = float('inf')
    orig_max_x = float('-inf')
    orig_max_y = float('-inf')
    
    for piece in original_pieces:
        if 'polygon' in piece:
            piece_min_x, piece_min_y, piece_max_x, piece_max_y = piece['polygon'].bounds
            orig_min_x = min(orig_min_x, piece_min_x)
            orig_min_y = min(orig_min_y, piece_min_y)
            orig_max_x = max(orig_max_x, piece_max_x)
            orig_max_y = max(orig_max_y, piece_max_y)
    
    original_width_used = orig_max_x - orig_min_x
    original_height_used = orig_max_y - orig_min_y
    original_area = original_width_used * original_height_used
    
    # For optimized layout, use the actual width and height we just calculated
    optimized_width_used = actual_width
    optimized_height_used = actual_height
    optimized_area = optimized_width_used * optimized_height_used
    
    # Create scrap analyzers for both layouts
    original_analyzer = ScrapAnalyzer(original_width, original_height)
    optimized_analyzer = ScrapAnalyzer(final_fabric_width, final_fabric_height)
    
    # Analyze both layouts
    original_analysis = original_analyzer.analyze(original_pieces)
    optimized_analysis = optimized_analyzer.analyze(pattern_pieces)
    
    # Extract metrics
    total_pattern_area = sum(piece['area'] for piece in pattern_pieces if 'new_x' in piece)
    original_reusable_area = original_analysis['total_usable_area']
    optimized_reusable_area = optimized_analysis['total_usable_area']
    
    # Calculate net fabric used (total area - reusable area)
    original_net_fabric_used = original_area - original_reusable_area
    optimized_net_fabric_used = optimized_area - optimized_reusable_area
    
    # Area reduction calculation - compare net fabric used
    area_reduction = original_net_fabric_used - optimized_net_fabric_used
    area_reduction_percent = (area_reduction / original_net_fabric_used) * 100 if original_net_fabric_used > 0 else 0
    
    # Calculate reusable ratios
    original_reusable_ratio = original_reusable_area / original_area if original_area > 0 else 0
    optimized_reusable_ratio = optimized_reusable_area / optimized_area if optimized_area > 0 else 0
    
    # Calculate improvement metrics
    reusable_improvement = optimized_reusable_ratio - original_reusable_ratio
    reusable_improvement_percent = (reusable_improvement / original_reusable_ratio) * 100 if original_reusable_ratio > 0 else 0
    
    # Count rotated pieces
    rotated_pieces = sum(1 for piece in pattern_pieces if piece.get('rotated', False))
    
    # Prepare statistics
    stats = [
        "**== Overall Fabric Required ==**",
        f"Fixed fabric dimensions: {final_fabric_width} × {final_fabric_height} pixels = {final_fabric_width * final_fabric_height} sq. pixels",
        f"Original layout used area: {int(original_width_used)} × {int(original_height_used)} pixels = {int(original_area)} sq. pixels",
        f"Optimized layout used area: {int(optimized_width_used)} × {int(optimized_height_used)} pixels = {int(optimized_area)} sq. pixels",
        
        "**== Net Fabric Consumption ==**",
        f"Original net fabric used: {int(original_net_fabric_used)} sq. pixels",
        f"Optimized net fabric used: {int(optimized_net_fabric_used)} sq. pixels",
        f"Actual fabric saving: {int(area_reduction)} sq. pixels ({area_reduction_percent:.2f}%)",
        f"Seam allowance applied: {margin} pixels",
        
        "**== Original Layout ==**",
        f"Original reusable fabric: {int(original_reusable_area)} sq. pixels",
        f"Original reusable fabric ratio: {original_reusable_ratio * 100:.2f}%",
        
        "**== Optimized Layout ==**",
        f"Optimized reusable fabric: {int(optimized_reusable_area)} sq. pixels",
        f"Optimized reusable fabric ratio: {optimized_reusable_ratio * 100:.2f}%",
        
        "**== Zero-Waste Metrics ==**",
        f"Total reusable fabric gained: {int(optimized_reusable_area - original_reusable_area)} sq. pixels",
        f"Reusable fabric improvement: {reusable_improvement_percent:.2f}%",
        
        "**== Other Stats ==**",
        f"Pieces placed: {placed_pieces}/{total_pieces}",
        f"Pieces rotated: {rotated_pieces}/{total_pieces}",
        f"Processing time: {time.time() - start_time:.2f} seconds",
        
        "**== Optimization Summary ==**",
        f"This optimization {'REDUCES' if area_reduction > 0 else 'INCREASES'} total fabric needed by {abs(area_reduction_percent):.2f}%, which {'saves' if area_reduction > 0 else 'increases'} material costs.",
        f"The reusable fabric quality {'improved' if reusable_improvement > 0 else 'decreased'} by {abs(reusable_improvement_percent):.1f}%."
    ]

    # Create output files
    svg_path = f"{output_prefix}.svg"
    pdf_path = f"{output_prefix}.pdf"
    
    # Create SVG
    create_svg(pattern_pieces, final_fabric_width, final_fabric_height, svg_path, margin)
    
    # Create PDF with both layouts and green rectangles
    create_pdf(svg_path, original_width, original_height, final_fabric_width, final_fabric_height,
               pattern_pieces, original_pieces, original_analysis, optimized_analysis, stats, pdf_path, margin)
    
    return {
        'pattern_pieces': pattern_pieces,
        'fabric_width': final_fabric_width,
        'fabric_height': final_fabric_height,
        'optimized_reusable_ratio': optimized_reusable_ratio,
        'original_reusable_ratio': original_reusable_ratio,
        'area_reduction_percent': area_reduction_percent,
        'rotated_pieces': rotated_pieces,
        'optimized_reusable_area': optimized_reusable_area,
        'original_reusable_area': original_reusable_area,
        'reusable_improvement': reusable_improvement_percent
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pattern Piece Optimizer with No-Fit Polygon')
    parser.add_argument('input_pdf', help='Path to input PDF with pattern pieces')
    parser.add_argument('--output', '-o', default='nfp_optimized_layout', help='Output file prefix')
    parser.add_argument('--width', '-w', type=int, help='Maximum fabric width in pixels')
    parser.add_argument('--rotation', '-r', action='store_true', default=True, help='Enable pattern piece rotation (default: True)')
    parser.add_argument('--no-rotation', action='store_true', default=False, help='Disable pattern piece rotation')
    parser.add_argument('--arbitrary-rotation', '-ar', action='store_true', default=False, 
                        help='Enable arbitrary rotation angles (not just 90 degrees)')
    parser.add_argument('--rotation-step', '-rs', type=int, default=15, 
                        help='Step size for rotation angles in degrees (default: 15)')
    parser.add_argument('--margin', '-m', type=int, default=5, help='Margin between pattern pieces in pixels (default: 5)')
    parser.add_argument('--max-positions', '-p', type=int, default=500,
                        help='Maximum number of positions to evaluate per piece (default: 500)')
    
    args = parser.parse_args()
    
    # Check if Shapely is installed, and provide helpful message if not
    try:
        import shapely
    except ImportError:
        print("ERROR: This program requires the 'shapely' package for NFP calculation and collision detection.")
        print("Please install it with: pip install shapely")
        exit(1)
    
    # Override rotation setting if --no-rotation is specified
    allow_rotation = args.rotation and not args.no_rotation
    
    # Pass the rotation settings to the optimize function
    optimize_pattern_layout(args.input_pdf, args.output, args.width, allow_rotation, 
                           args.arbitrary_rotation, args.margin, args.rotation_step)