import os
import numpy as np
import cv2
import argparse
import svgwrite
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A2
from reportlab.lib.units import mm
import time
from shapely.geometry import Polygon

# Constants for scrap analysis
MAX_RECTANGLE_COUNT = 10
LARGE_MASK_THRESHOLD = 1000

class ScrapAnalyzer:
    """Analyzer for fabric scrap focusing on key metrics."""
    
    def __init__(self, fabric_width, fabric_height):
        self.fabric_width = int(fabric_width)
        self.fabric_height = int(fabric_height)
        
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
        
        # Find the smallest piece's area (for threshold calculation)
        if pattern_pieces:
            smallest_piece_area = min(piece['area'] for piece in pattern_pieces)
            # Set minimum area threshold (half of smallest piece)
            min_usable_area = max(smallest_piece_area * 2.0, 1000)
        else:
            min_usable_area = 1000  # Default if no pieces
                
        # Find multiple usable rectangles
        usable_rectangles = self._find_multiple_rectangles(empty_mask, min_usable_area, max_count=MAX_RECTANGLE_COUNT)
        
        # Calculate largest rectangle (first in the list, if any)
        if usable_rectangles:
            max_rect_area = usable_rectangles[0]['area']
            max_rect_coords = usable_rectangles[0]['coords']
        else:
            max_rect_area = 0
            max_rect_coords = (0, 0, 0, 0)
                
        # Calculate total area of usable rectangles
        total_usable_area = sum(rect['area'] for rect in usable_rectangles)
        
        # Calculate scrap quality index
        scrap_quality_index = (total_usable_area / total_scrap_area) * 100 if total_scrap_area > 0 else 0
        
        # Count the total number of connected regions
        num_labels, _ = cv2.connectedComponents(empty_mask.astype(np.uint8), connectivity=8)
        scrap_regions = max(0, num_labels - 1)  # Subtract one for background
        
        # Calculate usable fabric ratio
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
                # Get adjusted points considering position
                points = piece['points']
                dx = float(piece['new_x'] - piece['x'])
                dy = float(piece['new_y'] - piece['y'])
                
                # Just translate
                adjusted_points = [(int(p[0] + dx), int(p[1] + dy)) for p in points]
                
                # Convert to numpy array for cv2
                adjusted_points_array = np.array(adjusted_points, dtype=np.int32)
                
                # Check if all points are within bounds to avoid index errors
                in_bounds = (
                    np.all(adjusted_points_array[:, 0] >= 0) and 
                    np.all(adjusted_points_array[:, 0] < self.fabric_width) and
                    np.all(adjusted_points_array[:, 1] >= 0) and 
                    np.all(adjusted_points_array[:, 1] < self.fabric_height)
                )
                
                if in_bounds:
                    # Fill polygon in the mask (0 = occupied)
                    cv2.fillPoly(mask, [adjusted_points_array], 0)
                else:
                    # Try to clip the polygon to fit within bounds
                    clipped_points = []
                    for px, py in adjusted_points:
                        clipped_points.append((
                            max(0, min(px, self.fabric_width - 1)),
                            max(0, min(py, self.fabric_height - 1))
                        ))
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
            # Calculate scaled dimensions
            scaled_x = int(x * scale_w)
            scaled_y = int(y * scale_h)
            scaled_w = int(w * scale_w)
            scaled_h = int(h * scale_h)
            
            # Check bounds after scaling
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
    
    def _find_multiple_rectangles(self, empty_mask, min_area=1000, max_count=MAX_RECTANGLE_COUNT):
        """Find multiple large rectangles in the empty space with optimized approach."""
        # Quick check for empty mask
        if np.sum(empty_mask) == 0:
            return []
            
        # Get mask dimensions for bounds checking
        height, width = empty_mask.shape
        
        # Make a copy of the original mask
        original_mask = empty_mask.copy()
        rectangles = []
        
        # Find rectangles iteratively with strict bounds checking
        for i in range(max_count):
            # Create a fresh working copy for this iteration
            mask = original_mask.copy()
            
            # If we've already found rectangles, mask them out
            for rect in rectangles:
                x, y, w, h = rect['coords']
                # Ensure coordinates are within bounds before masking
                x_start = max(0, x)
                y_start = max(0, y)
                x_end = min(width, x + w)
                y_end = min(height, y + h)
                
                if x_start < x_end and y_start < y_end:  # Only mask if region is valid
                    mask[y_start:y_end, x_start:x_end] = 0
            
            # Find largest rectangle in current mask
            rect_area, rect_coords = self._find_largest_rectangle(mask)
            x, y, w, h = rect_coords
            
            # Skip invalid rectangles with strict validation
            if (rect_area <= 0 or w <= 0 or h <= 0 or 
                x < 0 or y < 0 or 
                x + w > width or y + h > height or
                rect_area < min_area):
                break  # Stop if we can't find a valid rectangle
            
            # Double-check bounds for extra safety
            if not (0 <= x < width and 0 <= y < height and x + w <= width and y + h <= height):
                break
                
            # Add rectangle to list only if completely valid
            rectangles.append({
                'area': rect_area,
                'coords': rect_coords,
                'value_index': i
            })
            
            # Early exit if remaining area is less than min_area
            remaining_area = np.sum(mask) - rect_area
            if remaining_area < min_area:
                break
        
        return rectangles

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
        
        # Calculate bounding box for bin packing
        x, y, w, h = cv2.boundingRect(approx)
        
        # Create pattern piece object
        piece = {
            'name': f'piece_{i}',
            'points': points,
            'width': w,
            'height': h,
            'x': x,  # original x position
            'y': y,  # original y position
            'area': cv2.contourArea(approx)
        }
        
        pattern_pieces.append(piece)
    
    print(f"Extracted {len(pattern_pieces)} pattern pieces")
    return pattern_pieces, img.shape[1], img.shape[0]

class BinPacker:
    """Binary tree bin packing algorithm."""
    
    def __init__(self, max_width):
        self.max_width = max_width
        self.root = None
    
    def fit(self, blocks):
        """Fit blocks into the bin."""
        if not blocks:
            return
        
        # Sort blocks by height (tallest first)
        blocks.sort(key=lambda b: b['height'], reverse=True)
        
        # Initialize with first block's dimensions
        self.root = {"x": 0, "y": 0, "width": blocks[0]['width'], "height": blocks[0]['height'], "used": False}
        
        for block in blocks:
            # Find node to place this block
            node = self.find_node(self.root, block['width'], block['height'])
            if node:
                # If found, split the node and update block position
                fit = self.split_node(node, block['width'], block['height'])
                block['new_x'] = fit["x"]
                block['new_y'] = fit["y"]
            else:
                # If not found, grow the bin and try again
                fit = self.grow_node(block['width'], block['height'])
                if fit:
                    block['new_x'] = fit["x"]
                    block['new_y'] = fit["y"]
                else:
                    print(f"Could not place block {block['name']}")
    
    def find_node(self, root, width, height):
        """Find a node where the block can fit."""
        if not root:
            return None
            
        if root.get("used"):
            # Try right branch first, then down branch
            return self.find_node(root.get("right"), width, height) or self.find_node(root.get("down"), width, height)
        elif width <= root["width"] and height <= root["height"]:
            # Block fits in this node
            return root
        else:
            # Block doesn't fit
            return None
    
    def split_node(self, node, width, height):
        """Split a node after placing a block."""
        node["used"] = True
        
        # Create down node (space below the placed block)
        node["down"] = {
            "x": node["x"],
            "y": node["y"] + height,
            "width": node["width"],
            "height": node["height"] - height
        }
        
        # Create right node (space to the right of the placed block)
        node["right"] = {
            "x": node["x"] + width,
            "y": node["y"],
            "width": node["width"] - width,
            "height": height
        }
        
        return node
    
    def grow_node(self, width, height):
        """Grow the bin to accommodate a block that doesn't fit."""
        # Check if we can grow right (within max width)
        can_grow_right = (self.root["width"] + width) <= self.max_width
        # Can always grow down
        can_grow_down = True
        
        # Choose growth direction
        if can_grow_right and self.root["height"] >= (self.root["width"] + width):
            return self.grow_right(width, height)
        elif can_grow_down:
            return self.grow_down(width, height)
        elif can_grow_right:
            return self.grow_right(width, height)
        else:
            return None
    
    def grow_right(self, width, height):
        """Grow the bin to the right."""
        # Create new root node with increased width
        self.root = {
            "used": True,
            "x": 0,
            "y": 0,
            "width": self.root["width"] + width,
            "height": self.root["height"],
            "down": self.root,  # Old root becomes the down child
            "right": {  # New space to the right
                "x": self.root["width"],
                "y": 0,
                "width": width,
                "height": self.root["height"]
            }
        }
        
        # Try to find a node for the block in the enlarged bin
        node = self.find_node(self.root, width, height)
        if node:
            return self.split_node(node, width, height)
        return None
    
    def grow_down(self, width, height):
        """Grow the bin downward."""
        # Create new root node with increased height
        self.root = {
            "used": True,
            "x": 0,
            "y": 0,
            "width": self.root["width"],
            "height": self.root["height"] + height,
            "down": {  # New space at the bottom
                "x": 0,
                "y": self.root["height"],
                "width": self.root["width"],
                "height": height
            },
            "right": self.root  # Old root becomes the right child
        }
        
        # Try to find a node for the block in the enlarged bin
        node = self.find_node(self.root, width, height)
        if node:
            return self.split_node(node, width, height)
        return None

def trim_empty_space(pattern_pieces, fabric_width):
    """Find the actual height used by pattern pieces, trimming unused space."""
    if not pattern_pieces:
        return 0

    # Find the maximum y coordinate
    max_y = 0
    for piece in pattern_pieces:
        if 'new_y' in piece and 'height' in piece:
            max_y = max(max_y, piece['new_y'] + piece['height'])
    
    return max_y

def create_svg(pattern_pieces, fabric_width, fabric_height, output_path):
    """Create SVG visualization of the pattern layout."""
    # Create SVG drawing
    dwg = svgwrite.Drawing(output_path, profile='tiny')
    dwg.viewbox(0, 0, fabric_width, fabric_height)
    
    # Add fabric background
    dwg.add(dwg.rect((0, 0), (fabric_width, fabric_height), fill='white', stroke='black', stroke_width=1))
    
    # Color palette for pieces
    colors = ['#FF5555', '#55FF55', '#5555FF', '#FFFF55', '#FF55FF', '#55FFFF', 
              '#AA5555', '#55AA55', '#5555AA', '#AAAA55', '#AA55AA', '#55AAAA']
    
    # Add pattern pieces
    for i, piece in enumerate(pattern_pieces):
        if 'new_x' not in piece or 'new_y' not in piece:
            continue
            
        color = colors[i % len(colors)]
        
        # Calculate position offset to move points relative to new position
        dx = piece['new_x'] - piece['x']
        dy = piece['new_y'] - piece['y']
        
        # Adjust points to new position
        adjusted_points = [(p[0] + dx, p[1] + dy) for p in piece['points']]
        
        # Add polygon
        dwg.add(dwg.polygon(adjusted_points, fill=color, stroke='black', stroke_width=1))
        
        # Add label with piece name
        # Calculate centroid (average of points)
        x_coords = [p[0] for p in adjusted_points]
        y_coords = [p[1] for p in adjusted_points]
        centroid_x = sum(x_coords) / len(adjusted_points)
        centroid_y = sum(y_coords) / len(adjusted_points)
        
        dwg.add(dwg.text(piece['name'], insert=(centroid_x, centroid_y), 
                         text_anchor='middle', font_size='12'))
    
    # Save drawing
    dwg.save()
    print(f"SVG saved to {output_path}")
    
    return output_path

def calculate_original_fabric_area(pattern_pieces):
    """Calculate the minimum rectangle that contains all pattern pieces in original layout."""
    if not pattern_pieces:
        return 0, 0, 0, 0, 0
    
    # Calculate bounding box of all pieces
    min_x = min(min(p[0] for p in piece['points']) for piece in pattern_pieces)
    min_y = min(min(p[1] for p in piece['points']) for piece in pattern_pieces)
    max_x = max(max(p[0] for p in piece['points']) for piece in pattern_pieces)
    max_y = max(max(p[1] for p in piece['points']) for piece in pattern_pieces)
    
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
    
    return min_x, min_y, width, height, area

def calculate_smallest_enclosing_rectangle(pattern_pieces):
    """Calculate the smallest rectangle that contains all pattern pieces."""
    if not pattern_pieces:
        return 0, 0
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = 0
    max_y = 0
    
    for piece in pattern_pieces:
        if 'new_x' not in piece or 'new_y' not in piece:
            continue
            
        # Get points with position offset applied
        dx = piece['new_x'] - piece['x']
        dy = piece['new_y'] - piece['y']
        adjusted_points = [(p[0] + dx, p[1] + dy) for p in piece['points']]
        
        # Update min and max coordinates
        for px, py in adjusted_points:
            min_x = min(min_x, px)
            min_y = min(min_y, py)
            max_x = max(max_x, px)
            max_y = max(max_y, py)
    
    # Handle edge case
    if min_x == float('inf'):
        return 0, 0
        
    # Add margin for clarity
    margin = 10
    width = max(0, max_x - min_x + margin*2)
    height = max(0, max_y - min_y + margin*2)
    
    return width, height

def create_pdf(pattern_pieces, original_width, original_height, fabric_width, fabric_height,
               stats, original_analysis, optimized_analysis, output_pdf):
    """Create PDF with original and optimized layouts plus statistics."""
    # Use larger page size
    page_width, page_height = A2
    c = canvas.Canvas(output_pdf, pagesize=(page_width, page_height))
    
    # Calculate margins and layout positioning
    margin = 20 * mm
    available_height = page_height - 2 * margin
    available_width = page_width - 2 * margin
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, page_height - margin, "Bin Packing Pattern Piece Optimization")
    
    # Statistics section
    c.setFont("Helvetica", 12)
    y_pos = page_height - margin - 20 * mm
    
    # Draw comparison statistics by section
    section = None
    in_section = False
    
    for line in stats:
        # Handle section headers
        if line.startswith("=="):
            # End previous section if needed
            if in_section:
                c.restoreState()
                in_section = False
            
            # Start new section
            section = line.strip("= ")
            c.saveState()
            in_section = True
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y_pos, line)
            y_pos -= 6 * mm
            c.setFont("Helvetica", 10)  # Reset font for content after header
            continue
        
        # Handle empty lines
        if not line:
            y_pos -= 3 * mm
            continue
        
        # Draw regular stat line
        c.drawString(margin, y_pos, line)
        y_pos -= 5 * mm
    
    # End last section if still active
    if in_section:
        c.restoreState()
        in_section = False
    
    # Dividing line
    y_pos -= 5 * mm
    c.line(margin, y_pos, page_width - margin, y_pos)
    y_pos -= 10 * mm
    
    # Calculate original layout bounds
    orig_min_x, orig_min_y, orig_width, orig_height, _ = calculate_original_fabric_area(pattern_pieces)
    
    # Calculate max height needed for displays
    layout_height = max(orig_height, fabric_height)
    
    # Scale factors to fit layouts side by side
    max_layout_width = (available_width - 20 * mm) / 2
    max_layout_height = available_height - (page_height - y_pos)
    
    scale_original = min(max_layout_width / orig_width, max_layout_height / layout_height) if orig_width > 0 and layout_height > 0 else 1
    scale_optimized = min(max_layout_width / fabric_width, max_layout_height / layout_height) if fabric_width > 0 and layout_height > 0 else 1
    
    # Section headers
    c.setFont("Helvetica-Bold", 14)
    left_x = margin + max_layout_width/2
    right_x = margin + max_layout_width + 20*mm + max_layout_width/2
    c.drawString(left_x - 40*mm, y_pos, "Original Layout")
    c.drawString(right_x - 40*mm, y_pos, "Optimized Layout")
    y_pos -= 8 * mm
    
    # Calculate starting positions for layouts
    orig_start_y = y_pos - scale_original * layout_height
    optim_start_y = y_pos - scale_optimized * layout_height
    
    # Draw original layout rectangle
    original_layout_x = margin
    original_layout_y = orig_start_y
    c.rect(original_layout_x, original_layout_y, 
           scale_original * orig_width, scale_original * layout_height, 
           stroke=1, fill=0)
    
    # Draw optimized layout rectangle
    optimized_layout_x = margin + max_layout_width + 20*mm
    optimized_layout_y = optim_start_y
    c.rect(optimized_layout_x, optimized_layout_y, 
           scale_optimized * fabric_width, scale_optimized * fabric_height, 
           stroke=1, fill=0)
    
    # Draw original layout
    c.saveState()
    c.translate(original_layout_x - scale_original * orig_min_x, 
                original_layout_y - scale_original * orig_min_y)
    c.scale(scale_original, scale_original)
    
    # Draw original pieces
    for i, piece in enumerate(pattern_pieces):
        # Set piece color (rotate through colors)
        r, g, b = [(i % 3 == 0) * 0.8, (i % 3 == 1) * 0.8, (i % 3 == 2) * 0.8]
        c.setFillColorRGB(r, g, b)
        
        # Draw polygon
        path = c.beginPath()
        path.moveTo(piece['points'][0][0], piece['points'][0][1])
        for point in piece['points'][1:]:
            path.lineTo(point[0], point[1])
        path.close()
        c.drawPath(path, fill=1, stroke=1)
    
    # Draw original usable rectangles
    if original_analysis and 'usable_rectangles' in original_analysis:
        for i, rect_data in enumerate(original_analysis['usable_rectangles']):
            rect = rect_data['coords']
            x, y, w, h = rect
            
            # Only draw valid rectangles
            if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= orig_width and y + h <= orig_height):
                # Vary shade of green for different rectangles
                green_level = max(0.3, 0.8 - i * 0.1)
                c.setStrokeColorRGB(0, 0.8, 0)
                c.setFillColorRGB(0, green_level, 0, 0.2)
                c.setDash(5, 5)
                c.rect(x, y, w, h, fill=1, stroke=1)
    
    c.restoreState()
    
    # Draw optimized layout
    c.saveState()
    c.translate(optimized_layout_x, optimized_layout_y)
    c.scale(scale_optimized, scale_optimized)
    
    # Draw optimized pieces
    for i, piece in enumerate(pattern_pieces):
        if 'new_x' not in piece or 'new_y' not in piece:
            continue
            
        # Set piece color
        r, g, b = [(i % 3 == 0) * 0.8, (i % 3 == 1) * 0.8, (i % 3 == 2) * 0.8]
        c.setFillColorRGB(r, g, b)
        
        # Calculate position offset
        dx = piece['new_x'] - piece['x']
        dy = piece['new_y'] - piece['y']
        
        # Adjust points to new position
        adjusted_points = [(p[0] + dx, p[1] + dy) for p in piece['points']]
        
        # Draw polygon
        path = c.beginPath()
        path.moveTo(adjusted_points[0][0], adjusted_points[0][1])
        for point in adjusted_points[1:]:
            path.lineTo(point[0], point[1])
        path.close()
        c.drawPath(path, fill=1, stroke=1)
    
    # Draw optimized usable rectangles
    if optimized_analysis and 'usable_rectangles' in optimized_analysis:
        for i, rect_data in enumerate(optimized_analysis['usable_rectangles']):
            rect = rect_data['coords']
            x, y, w, h = rect
            
            # Only draw valid rectangles
            if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= fabric_width and y + h <= fabric_height):
                # Vary shade of green for different rectangles
                green_level = max(0.3, 0.8 - i * 0.1)
                c.setStrokeColorRGB(0, 0.8, 0)
                c.setFillColorRGB(0, green_level, 0, 0.2)
                c.setDash(5, 5)
                c.rect(x, y, w, h, fill=1, stroke=1)
    
    c.restoreState()
    
    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(margin, margin/2, "Green highlighted rectangles represent reusable fabric pieces")
    
    # Save the PDF
    c.save()
    print(f"PDF saved to {output_pdf}")

def optimize_pattern_layout(input_pdf, output_prefix, fabric_width=None):
    """Main function to optimize pattern layout."""
    start_time = time.time()
    
    # Extract pattern pieces from PDF
    pattern_pieces, original_width, original_height = extract_pattern_pieces(input_pdf)
    
    if not pattern_pieces:
        print("No pattern pieces were extracted.")
        return
    
    # Calculate original layout area - the actual used area
    orig_min_x, orig_min_y, orig_width, orig_height, original_fabric_area = calculate_original_fabric_area(pattern_pieces)
    
    # Create a copy of pattern pieces in original positions
    original_pieces = []
    for piece in pattern_pieces:
        original_piece = piece.copy()
        original_piece['new_x'] = original_piece['x']
        original_piece['new_y'] = original_piece['y']
        original_pieces.append(original_piece)
    
    # Analyze the original layout for baseline metrics
    print("Analyzing original layout...")
    original_analyzer = ScrapAnalyzer(orig_width, orig_height)
    original_analysis = original_analyzer.analyze(original_pieces)
    
    # Use a reasonable fabric width if not provided
    if fabric_width is None:
        # Use original width or 90% of it if it's very large
        fabric_width = min(orig_width, int(original_width * 0.9))
    
    print(f"Using fabric width: {fabric_width}")
    
    # Create the bin packer
    packer = BinPacker(fabric_width)
    
    # Run bin packing
    packer.fit(pattern_pieces)
    
    # Calculate actual height used and trim unused space
    fabric_height = trim_empty_space(pattern_pieces, fabric_width)
    
    # Calculate the actual used area after optimization
    actual_used_width, actual_used_height = calculate_smallest_enclosing_rectangle(pattern_pieces)
    actual_used_area = actual_used_width * actual_used_height
    
    # Analyze scraps in optimized layout
    print("Analyzing optimized layout...")
    optimized_analyzer = ScrapAnalyzer(fabric_width, fabric_height)
    optimized_analysis = optimized_analyzer.analyze(pattern_pieces)
    
    # Calculate statistics
    total_pattern_area = sum(piece['area'] for piece in pattern_pieces)
    
    # Before optimization
    original_utilization = (total_pattern_area / original_fabric_area) * 100 if original_fabric_area > 0 else 0
    
    # After optimization
    optimized_fabric_area = fabric_width * fabric_height
    optimized_utilization = (total_pattern_area / optimized_fabric_area) * 100 if optimized_fabric_area > 0 else 0
    
    # Calculate fabric savings
    fabric_saved = original_fabric_area - actual_used_area
    fabric_saved_percent = (fabric_saved / original_fabric_area) * 100 if original_fabric_area > 0 else 0
    
    # Calculate usable fabric metrics
    total_usable_fabric_gained = float(optimized_analysis['total_usable_area']) - float(original_analysis['total_usable_area'])
    quality_improvement = optimized_analysis['scrap_quality_index'] - original_analysis['scrap_quality_index']
    
    # Calculate net fabric consumption
    original_net_fabric_used = original_fabric_area - original_analysis['total_usable_area']
    optimized_net_fabric_used = actual_used_area - optimized_analysis['total_usable_area']
    net_fabric_reduction = original_net_fabric_used - optimized_net_fabric_used
    net_fabric_reduction_percent = (net_fabric_reduction / original_net_fabric_used) * 100 if original_net_fabric_used > 0 else 0
    
    # Count placed pieces
    placed_count = sum(1 for piece in pattern_pieces if 'new_x' in piece)
    
    # Format statistics in requested format
    stats = [
        "== Overall Fabric Required ==",
        f"Fixed fabric dimensions: {fabric_width} × {fabric_height} pixels = {fabric_width * fabric_height:,} sq. pixels",
        f"Original layout used area: {orig_width} × {orig_height} pixels = {original_fabric_area:,} sq. pixels",
        f"Optimized layout used area: {actual_used_width} × {actual_used_height} pixels = {actual_used_area} sq. pixels",
        f"Actual area reduction: {fabric_saved:,} sq. pixels ({fabric_saved_percent:.2f}%)",
        "Seam allowance applied: 0 pixels",
        "",
        "== Net Fabric Consumption ==",
        f"Original net fabric used: {original_net_fabric_used:,} sq. pixels",
        f"Optimized net fabric used: {optimized_net_fabric_used:,} sq. pixels",
        f"Net fabric reduction: {net_fabric_reduction:,} sq. pixels ({net_fabric_reduction_percent:.2f}%)",
        "",
        "== Original Layout ==",
        f"Original reusable fabric: {original_analysis['total_usable_area']:,} sq. pixels",
        f"Original reusable fabric ratio: {original_analysis['scrap_quality_index']:.2f}%",
        "",
        "== Optimized Layout ==",
        f"Optimized reusable fabric: {optimized_analysis['total_usable_area']:,} sq. pixels",
        f"Optimized reusable fabric ratio: {optimized_analysis['scrap_quality_index']:.2f}%",
        "",
        "== Zero-Waste Metrics ==",
        f"Total reusable fabric gained: {total_usable_fabric_gained:.2f} sq. pixels",
        f"Reusable fabric improvement: {quality_improvement:.2f}%",
        "",
        "== Other Stats ==",
        f"Pieces placed: {placed_count}/{len(pattern_pieces)}",
        f"Processing time: {time.time() - start_time:.2f} seconds",
        "",
        "== Optimization Summary =="
    ]
    
    # Add optimization summary
    if net_fabric_reduction_percent > 0:
        stats.append(f"This optimization REDUCES total fabric needed by {net_fabric_reduction_percent:.2f}%, which saves material costs.")
    else:
        stats.append(f"This optimization INCREASES total fabric area by {abs(net_fabric_reduction_percent):.2f}%, but provides more reusable leftover fabric.")
        
    stats.append(f"The reusable fabric quality improved by {quality_improvement:.2f}%")
    
    # Print statistics
    print("\n".join(stats))
    
    # Create output files
    svg_path = f"{output_prefix}.svg"
    pdf_path = f"{output_prefix}.pdf"
    
    # Create SVG (no rectangles in visualization)
    create_svg(pattern_pieces, fabric_width, fabric_height, svg_path)
    
    # Create PDF (with green rectangles for usable space)
    create_pdf(pattern_pieces, original_width, original_height, fabric_width, fabric_height,
               stats, original_analysis, optimized_analysis, pdf_path)
    
    return {
        'pattern_pieces': pattern_pieces,
        'fabric_width': fabric_width,
        'fabric_height': fabric_height,
        'original_fabric_area': original_fabric_area,
        'optimized_fabric_area': optimized_fabric_area,
        'total_pattern_area': total_pattern_area,
        'original_utilization': original_utilization,
        'optimized_utilization': optimized_utilization,
        'fabric_saved_percent': fabric_saved_percent,
        'original_analysis': original_analysis,
        'optimized_analysis': optimized_analysis
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pattern Piece Optimizer')
    parser.add_argument('input_pdf', help='Path to input PDF with pattern pieces')
    parser.add_argument('--output', '-o', default='optimized_layout', help='Output file prefix')
    parser.add_argument('--width', '-w', type=int, help='Maximum fabric width in pixels')
    
    args = parser.parse_args()
    
    optimize_pattern_layout(args.input_pdf, args.output, args.width)