import os
import numpy as np
import cv2
import gym
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from shapely.geometry import Polygon, Point
import torch
import json
import svgwrite
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from pdf2image import convert_from_path
import argparse
from tqdm import tqdm
import re
import xml.etree.ElementTree as ET
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternPiece:
    """Class to represent a single pattern piece with its properties."""
    
    def __init__(self, name, points=None, polygon=None, grain_constraint="any"):
        """
        Initialize a pattern piece from points or polygon.
        
        Args:
            name: Name or identifier for the pattern piece
            points: List of points defining the pattern piece contour
            polygon: Shapely polygon representing the pattern piece
            grain_constraint: Fabric grain constraint ("warp", "weft", "bias", or "any")
        """
        self.name = name
        self.grain_constraint = grain_constraint
        self.position = (0, 0)
        self.rotation = 0
        
        if polygon is not None:
            self.polygon = polygon
        elif points is not None:
            self.polygon = Polygon(points)
        else:
            raise ValueError("Either points or polygon must be provided")
            
        self.bounds = self.polygon.bounds  # (minx, miny, maxx, maxy)
        self.area = self.polygon.area
    
    def get_rotated_polygon(self, angle=None):
        """Get the polygon rotated by the specified angle (or current rotation)."""
        if angle is None:
            angle = self.rotation
        return rotate(self.polygon, angle, origin='centroid')
    
    def get_transformed_polygon(self, position=None, angle=None):
        """Get the polygon rotated and translated to position."""
        if position is None:
            position = self.position
        if angle is None:
            angle = self.rotation
            
        rotated = self.get_rotated_polygon(angle)
        return translate(rotated, position[0], position[1])
    
    def get_allowed_rotations(self):
        """Get list of allowed rotation angles based on grain constraint."""
        if self.grain_constraint == "warp":
            return [0, 180]  # Only allow 0 and 180 degrees
        elif self.grain_constraint == "weft":
            return [90, 270]  # Only allow 90 and 270 degrees
        elif self.grain_constraint == "bias":
            return [45, 135, 225, 315]  # Only allow bias angles
        else:  # "any"
            return [0, 45, 90, 135, 180, 225, 270, 315]  # Allow all standard rotations
    
    def update_position(self, position, rotation):
        """Update the position and rotation of the pattern piece."""
        self.position = position
        self.rotation = rotation


class FreeSewingPatternLoader:
    """Class to load and parse FreeSewing pattern data."""
    
    def __init__(self, json_path=None, pdf_path=None):
        """
        Initialize the pattern loader.
        
        Args:
            json_path: Path to FreeSewing JSON file (optional)
            pdf_path: Path to PDF file containing pattern pieces (optional)
        """
        self.json_path = json_path
        self.pdf_path = pdf_path
        self.pattern_data = None
        self.measurements = None
        self.pieces = {}
        
        # Try to load data from provided paths
        if json_path and os.path.exists(json_path):
            self.load_json(json_path)
        
        if pdf_path and os.path.exists(pdf_path):
            self.extract_from_pdf(pdf_path)
    
    def load_json(self, json_path):
        """
        Load pattern data from FreeSewing JSON file.
        
        Args:
            json_path: Path to the JSON file
        """
        logger.info(f"Loading FreeSewing JSON from {json_path}")
        try:
            with open(json_path, 'r') as f:
                self.pattern_data = json.load(f)
                
            # Extract measurements
            if 'measurements' in self.pattern_data:
                self.measurements = self.pattern_data['measurements']
                logger.info(f"Loaded {len(self.measurements)} measurements")
                
            # Extract pattern pieces from layout
            if 'layout' in self.pattern_data and 'stacks' in self.pattern_data['layout']:
                self._parse_layout_stacks()
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
    
    def _parse_layout_stacks(self):
        """Parse pattern pieces from layout stacks in JSON data."""
        stacks = self.pattern_data['layout']['stacks']
        logger.info(f"Found {len(stacks)} pattern pieces in layout")
        
        for name, piece_data in stacks.items():
            try:
                # Extract actual pattern piece name (removing prefix)
                if '.' in name:
                    piece_name = name.split('.')[-1]
                else:
                    piece_name = name
                
                # Extract corner points
                points = []
                if 'tl' in piece_data and 'br' in piece_data:
                    # Create rectangle from top-left and bottom-right corners
                    tl = piece_data['tl']
                    br = piece_data['br']
                    
                    points = [
                        (tl['x'], tl['y']),
                        (br['x'], tl['y']),
                        (br['x'], br['y']),
                        (tl['x'], br['y'])
                    ]
                    
                    # Create pattern piece
                    piece = PatternPiece(piece_name, points=points)
                    
                    # Set position and rotation
                    if 'move' in piece_data:
                        piece.position = (piece_data['move']['x'], piece_data['move']['y'])
                    if 'rotate' in piece_data:
                        piece.rotation = piece_data['rotate']
                    
                    # Add to pieces dictionary
                    self.pieces[piece_name] = piece
                    
                    logger.info(f"Added piece {piece_name} with area {piece.area:.2f}")
                else:
                    logger.warning(f"Piece {name} missing corner points")
            except Exception as e:
                logger.warning(f"Error parsing piece {name}: {e}")
    
    def extract_from_pdf(self, pdf_path):
        """
        Extract pattern pieces from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
        """
        logger.info(f"Extracting pattern pieces from PDF: {pdf_path}")
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            logger.info(f"Extracted {len(images)} pages from PDF")
            
            # Process each page
            for i, img in enumerate(images):
                # Convert PIL image to OpenCV format
                open_cv_image = np.array(img)
                open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
                
                # Convert to grayscale
                gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold to get binary image
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter and process contours
                for j, contour in enumerate(contours):
                    # Filter small contours
                    if cv2.contourArea(contour) < 1000:
                        continue
                    
                    # Approximate the contour to reduce number of points
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Generate piece name
                    piece_name = f"piece_{i}_{j}"
                    
                    # Convert to list of tuples
                    points = [(point[0][0], point[0][1]) for point in approx]
                    
                    # Create pattern piece
                    piece = PatternPiece(piece_name, points=points)
                    
                    # Add to pieces dictionary
                    self.pieces[piece_name] = piece
                    
                    logger.info(f"Added piece {piece_name} with area {piece.area:.2f}")
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            raise
    
    def extract_from_svg(self, svg_path):
        """
        Extract pattern pieces from an SVG file.
        
        Args:
            svg_path: Path to the SVG file
        """
        logger.info(f"Extracting pattern pieces from SVG: {svg_path}")
        try:
            # Parse SVG file
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Find namespace
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            
            # Extract paths (pattern pieces)
            paths = root.findall('.//svg:path', ns)
            logger.info(f"Found {len(paths)} paths in SVG")
            
            for i, path in enumerate(paths):
                # Get path data
                d = path.get('d')
                if not d:
                    continue
                
                # Parse path data to extract points
                points = self._parse_svg_path(d)
                
                if len(points) < 3:  # Need at least 3 points for a polygon
                    continue
                
                # Generate piece name
                piece_name = path.get('id', f"piece_{i}")
                
                # Create pattern piece
                piece = PatternPiece(piece_name, points=points)
                
                # Add to pieces dictionary
                self.pieces[piece_name] = piece
                
                logger.info(f"Added piece {piece_name} with area {piece.area:.2f}")
        except Exception as e:
            logger.error(f"Error extracting from SVG: {e}")
            raise
    
    def _parse_svg_path(self, d):
        """
        Parse SVG path data to extract points.
        
        Args:
            d: Path data string
            
        Returns:
            List of (x, y) tuples
        """
        # This is a simplified parser that handles common path commands
        points = []
        current_point = (0, 0)
        
        # Split path data into commands
        commands = re.findall(r'([MLHVCSQTAZ])([^MLHVCSQTAZ]*)', d.upper())
        
        for cmd, params in commands:
            # Split parameters into numbers
            params = re.findall(r'-?\d+\.?\d*', params)
            params = [float(p) for p in params]
            
            if cmd == 'M':  # Move to
                for i in range(0, len(params), 2):
                    current_point = (params[i], params[i+1])
                    points.append(current_point)
            
            elif cmd == 'L':  # Line to
                for i in range(0, len(params), 2):
                    current_point = (params[i], params[i+1])
                    points.append(current_point)
            
            elif cmd == 'H':  # Horizontal line
                for p in params:
                    current_point = (p, current_point[1])
                    points.append(current_point)
            
            elif cmd == 'V':  # Vertical line
                for p in params:
                    current_point = (current_point[0], p)
                    points.append(current_point)
            
            # Add more command handlers as needed
            
        return points
    
    def apply_measurements(self):
        """
        Apply measurements to scale pattern pieces.
        This is a placeholder for more complex scaling logic.
        """
        if not self.measurements:
            logger.warning("No measurements available to apply")
            return
        
        logger.info("Applying measurements to pattern pieces")
        
        # For now, we just log the measurements
        for name, value in self.measurements.items():
            logger.info(f"Measurement: {name} = {value}")
        
        # In a more complete implementation, we would scale pattern pieces
        # based on the measurements and the pattern's grading rules


class FabricEnvironment(gym.Env):
    """Custom Environment for pattern piece placement optimization."""
    
    def __init__(self, pattern_pieces, fabric_width, max_length=None, 
                 grid_size=10, render_mode="rgb_array"):
        """
        Initialize the fabric environment.
        
        Args:
            pattern_pieces: List of PatternPiece objects
            fabric_width: Width of the fabric
            max_length: Maximum length of the fabric (or None to auto-calculate)
            grid_size: Size of each grid cell for placement discretization
            render_mode: How to render the environment
        """
        super().__init__()
        self.render_mode = render_mode
        
        self.pattern_pieces = pattern_pieces
        self.fabric_width = fabric_width
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Calculate a reasonable max_length if not provided
        if max_length is None:
            total_area = sum(piece.area for piece in pattern_pieces)
            # Add 50% padding to account for inefficient packing
            self.max_length = int(1.5 * total_area / fabric_width)
        else:
            self.max_length = max_length
            
        # Calculate grid dimensions
        self.grid_width = int(fabric_width / grid_size)
        self.grid_length = int(self.max_length / grid_size)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),  # x, y, rotation_index
            high=np.array([self.grid_width - 1, self.grid_length - 1, 7]),  # max x, y, rotation options
            shape=(3,),
            dtype=np.int32
        )
        
        # Observation space: fabric grid (binary mask) + one-hot encoding of current piece
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=(self.grid_width, self.grid_length), dtype=np.uint8),
            'current_piece': spaces.Discrete(len(pattern_pieces))
        })
        
        # Initialize environment state
        self.reset()
        
    def reset(self):
        """Reset the environment to the initial state."""
        # Reset the fabric grid (0 = empty, 1 = occupied)
        self.grid = np.zeros((self.grid_width, self.grid_length), dtype=np.uint8)
        
        # Reset pattern pieces
        for piece in self.pattern_pieces:
            piece.position = (0, 0)
            piece.rotation = 0
            
        # Track which pieces have been placed
        self.placed_pieces = []
        self.remaining_pieces = list(range(len(self.pattern_pieces)))
        
        # Select the first piece to place
        self.current_piece_idx = self.remaining_pieces.pop(0)
        self.current_piece = self.pattern_pieces[self.current_piece_idx]
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get the current observation state."""
        return {
            'grid': self.grid.copy(),
            'current_piece': self.current_piece_idx
        }
    
    def _is_valid_placement(self, piece, x, y, rotation):
        """Check if the placement is valid (within bounds and not overlapping)."""
        # Convert grid coordinates to actual coordinates
        x_pos = x * self.grid_size
        y_pos = y * self.grid_size
        
        # Get the transformed polygon
        transformed = piece.get_transformed_polygon((x_pos, y_pos), rotation)
        
        # Check if the piece is within the fabric bounds
        bounds = transformed.bounds  # (minx, miny, maxx, maxy)
        if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > self.fabric_width or bounds[3] > self.max_length:
            return False
        
        # Check if the piece overlaps with already placed pieces
        for placed_idx in self.placed_pieces:
            placed_piece = self.pattern_pieces[placed_idx]
            placed_polygon = placed_piece.get_transformed_polygon()
            if transformed.intersects(placed_polygon):
                return False
                
        return True
    
    def _update_grid(self):
        """Update the grid representation based on placed pieces."""
        self.grid = np.zeros((self.grid_width, self.grid_length), dtype=np.uint8)
        
        for idx in self.placed_pieces:
            piece = self.pattern_pieces[idx]
            transformed = piece.get_transformed_polygon()
            
            # Convert polygon to a mask
            minx, miny, maxx, maxy = transformed.bounds
            minx_grid = max(0, int(minx / self.grid_size))
            miny_grid = max(0, int(miny / self.grid_size))
            maxx_grid = min(self.grid_width, int(np.ceil(maxx / self.grid_size)))
            maxy_grid = min(self.grid_length, int(np.ceil(maxy / self.grid_size)))
            
            # Mark grid cells occupied by the piece
            for i in range(minx_grid, maxx_grid):
                for j in range(miny_grid, maxy_grid):
                    # Check if the center of this grid cell is inside the polygon
                    cell_center = (i * self.grid_size + self.grid_size/2, 
                                   j * self.grid_size + self.grid_size/2)
                    if transformed.contains(Point(cell_center)):
                        self.grid[i, j] = 1
    
    def step(self, action):
        """
        Take a step in the environment by placing the current piece.
        
        Args:
            action: Array [x, y, rotation_index]
            
        Returns:
            observation, reward, done, info
        """
        x, y, rotation_idx = action.astype(int)
        
        # Get allowed rotations for the current piece
        allowed_rotations = self.current_piece.get_allowed_rotations()
        rotation = allowed_rotations[rotation_idx % len(allowed_rotations)]
        
        # Check if the placement is valid
        if self._is_valid_placement(self.current_piece, x, y, rotation):
            # Update piece position and rotation
            self.current_piece.update_position((x * self.grid_size, y * self.grid_size), rotation)
            
            # Add to placed pieces
            self.placed_pieces.append(self.current_piece_idx)
            
            # Update grid
            self._update_grid()
            
            # Calculate the highest y-coordinate used (for length calculation)
            self.current_max_y = max([self.pattern_pieces[idx].get_transformed_polygon().bounds[3] 
                                      for idx in self.placed_pieces])
            
            reward = 1.0  # Base reward for successful placement
            
            # Add bonus for compactness (keeping pattern pieces close together)
            compactness_bonus = 1.0 - (self.current_max_y / self.max_length)
            reward += compactness_bonus
            
        else:
            # Invalid placement
            reward = -1.0
            
            # Add penalty based on how far the piece would go out of bounds
            transformed = self.current_piece.get_transformed_polygon(
                (x * self.grid_size, y * self.grid_size), rotation)
            bounds = transformed.bounds
            
            out_of_bounds_x = max(0, -bounds[0]) + max(0, bounds[2] - self.fabric_width)
            out_of_bounds_y = max(0, -bounds[1]) + max(0, bounds[3] - self.max_length)
            
            # Normalize by grid size
            out_of_bounds_penalty = (out_of_bounds_x + out_of_bounds_y) / (10 * self.grid_size)
            reward -= out_of_bounds_penalty
        
        # Move to the next piece if placement was valid
        if len(self.remaining_pieces) > 0 and reward > 0:
            self.current_piece_idx = self.remaining_pieces.pop(0)
            self.current_piece = self.pattern_pieces[self.current_piece_idx]
        
        # Check if all pieces have been placed
        done = len(self.placed_pieces) == len(self.pattern_pieces)
        
        # Add final reward based on fabric utilization if done
        if done:
            # Calculate total fabric area used
            used_length = self.current_max_y
            used_area = self.fabric_width * used_length
            
            # Sum the area of all pattern pieces
            total_pattern_area = sum(piece.area for piece in self.pattern_pieces)
            
            # Calculate utilization percentage
            utilization = total_pattern_area / used_area
            
            # Add bonus reward based on utilization
            utilization_bonus = 10.0 * utilization
            reward += utilization_bonus
        
        info = {
            'valid_placement': reward > 0,
            'utilization': self._calculate_utilization() if done else 0,
            'fabric_saving': self._calculate_saving() if done else 0
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_utilization(self):
        """Calculate the fabric utilization percentage."""
        if not self.placed_pieces:
            return 0
            
        # Calculate the highest y-coordinate used
        max_y = max([self.pattern_pieces[idx].get_transformed_polygon().bounds[3] 
                     for idx in self.placed_pieces])
                     
        # Calculate total fabric area used
        used_area = self.fabric_width * max_y
        
        # Sum the area of all pattern pieces
        total_pattern_area = sum(piece.area for piece in self.pattern_pieces)
        
        return total_pattern_area / used_area
    
    def _calculate_saving(self):
        """Calculate the fabric saving compared to traditional layout."""
        if not self.placed_pieces:
            return 0
            
        # Calculate the highest y-coordinate used
        optimized_length = max([self.pattern_pieces[idx].get_transformed_polygon().bounds[3] 
                                 for idx in self.placed_pieces])
        
        # Compare to FreeSewing's layout if available
        if hasattr(self, 'freesewing_layout_height'):
            traditional_length = self.freesewing_layout_height
        else:
            # Estimate traditional layout (naive row-by-row placement)
            traditional_length = 0
            row_width = 0
            
            # Sort pieces by height for a simple row-based layout
            sorted_pieces = sorted(self.pattern_pieces, key=lambda p: p.bounds[3] - p.bounds[1], reverse=True)
            
            for piece in sorted_pieces:
                width = piece.bounds[2] - piece.bounds[0]
                height = piece.bounds[3] - piece.bounds[1]
                
                # If piece doesn't fit in current row, start a new row
                if row_width + width > self.fabric_width:
                    traditional_length += height
                    row_width = width
                else:
                    row_width += width
            
            # Add the height of the last row
            if row_width > 0 and sorted_pieces:
                traditional_length += sorted_pieces[0].bounds[3] - sorted_pieces[0].bounds[1]
        
        # Calculate saving percentage
        saving = (traditional_length - optimized_length) / traditional_length
        return max(0, saving)  # Ensure non-negative saving
    
    def set_freesewing_layout_height(self, height):
        """Set the height of FreeSewing's layout for comparison."""
        self.freesewing_layout_height = height
    
    def render(self, mode='human'):
        """Render the current state of the environment."""
        mode = mode or "rgb_array"
        
        # Create a visual representation of the fabric with placed pieces
        scale = 5  # Scale for better visualization
        img_width = int(self.fabric_width * scale / self.grid_size)
        img_length = int(self.max_length * scale / self.grid_size)
        
        img = np.ones((img_length, img_width, 3), dtype=np.uint8) * 255
        
        # Draw grid lines
        for i in range(0, img_width, scale):
            cv2.line(img, (i, 0), (i, img_length), (200, 200, 200), 1)
        for j in range(0, img_length, scale):
            cv2.line(img, (0, j), (img_width, j), (200, 200, 200), 1)
        
        # Draw placed pieces
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        
        for i, idx in enumerate(self.placed_pieces):
            piece = self.pattern_pieces[idx]
            color = colors[i % len(colors)]
            
            # Get the transformed polygon
            polygon = piece.get_transformed_polygon()
            
            # Convert polygon to contour for OpenCV
            contour = np.array(list(polygon.exterior.coords)).astype(np.float32)
            
            # Scale for rendering
            scaled_contour = (contour * scale / self.grid_size).astype(np.int32)
            
            # Draw contour
            cv2.drawContours(img, [scaled_contour], 0, color, -1)
            
            # Draw outline
            cv2.drawContours(img, [scaled_contour], 0, (0, 0, 0), 2)
            
            # Draw label
            centroid = np.mean(scaled_contour, axis=0).astype(int)
            cv2.putText(img, piece.name, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        if mode == 'human':
            plt.figure(figsize=(12, 12))
            plt.imshow(img)
            plt.title('Fabric Layout')
            plt.show()
        
        return img

    def export_svg(self, output_path):
        """
        Export the current layout as an SVG file.
        
        Args:
            output_path: Path to save the SVG file
        """
        # Create an SVG drawing
        dwg = svgwrite.Drawing(output_path, profile='tiny', 
                              size=(f"{self.fabric_width}mm", f"{self.max_length}mm"))
        
        # Add fabric background
        dwg.add(dwg.rect(insert=(0, 0), size=(f"{self.fabric_width}mm", f"{self.max_length}mm"), 
                         fill='white', stroke='black', stroke_width=1))
        
        # Add grid lines (optional)
        for i in range(0, self.fabric_width, self.grid_size):
            dwg.add(dwg.line(start=(f"{i}mm", "0"), end=(f"{i}mm", f"{self.max_length}mm"), 
                             stroke='#eee', stroke_width=0.5))
        for j in range(0, self.max_length, self.grid_size):
            dwg.add(dwg.line(start=("0", f"{j}mm"), end=(f"{self.fabric_width}mm", f"{j}mm"), 
                             stroke='#eee', stroke_width=0.5))
        
        # Create a color palette for the pieces
        colors = [
            '#FF5555', '#55FF55', '#5555FF', 
            '#FFFF55', '#FF55FF', '#55FFFF',
            '#AA5555', '#55AA55', '#5555AA',
            '#AAAA55', '#AA55AA', '#55AAAA'
        ]
        
        # Add each placed piece
        for i, idx in enumerate(self.placed_pieces):
            piece = self.pattern_pieces[idx]
            color = colors[i % len(colors)]
            
            # Get the transformed polygon
            polygon = piece.get_transformed_polygon()
            
            # Create polygon points for SVG
            points = []
            for point in polygon.exterior.coords:
                points.append((f"{point[0]}mm", f"{point[1]}mm"))
            
            # Add polygon
            dwg.add(dwg.polygon(points=points, fill=color, stroke='black', stroke_width=1))
            
            # Add label with piece name
            centroid = polygon.centroid.coords[0]
            dwg.add(dwg.text(piece.name, insert=(f"{centroid[0]}mm", f"{centroid[1]}mm"), 
                             text_anchor='middle', font_size='8pt', font_family='Arial'))
        
        # Save the drawing
        dwg.save()
        logger.info(f"SVG exported to {output_path}")
        
        # Return metrics
        max_y = max([self.pattern_pieces[idx].get_transformed_polygon().bounds[3] 
                     for idx in self.placed_pieces]) if self.placed_pieces else 0
        
        return {
            'width': self.fabric_width,
            'length': max_y,
            'area': self.fabric_width * max_y
        }


class FreeSewingOptimizer:
    """Main class for optimizing FreeSewing patterns."""
    
    def __init__(self, json_path=None, pdf_path=None, svg_path=None, 
                 fabric_width=None, grid_size=10, model_path=None, log_dir='logs'):
        """
        Initialize the optimizer.
        
        Args:
            json_path: Path to FreeSewing JSON file
            pdf_path: Path to PDF file with pattern pieces
            svg_path: Path to SVG file with pattern pieces
            fabric_width: Width of the fabric in mm (or None to use value from FreeSewing)
            grid_size: Grid size for discretization
            model_path: Path to a pre-trained model (or None to train a new one)
            log_dir: Directory to save logs
        """
        self.json_path = json_path
        self.pdf_path = pdf_path
        self.svg_path = svg_path
        self.grid_size = grid_size
        self.model_path = model_path
        self.log_dir = log_dir
        
        # Initialize pattern loader
        self.loader = FreeSewingPatternLoader(json_path, pdf_path)
        
        # Load SVG if provided
        if svg_path:
            self.loader.extract_from_svg(svg_path)
        
        # Get pattern pieces
        self.pattern_pieces = list(self.loader.pieces.values())
        
        # Set fabric width
        if fabric_width:
            self.fabric_width = fabric_width
        elif self.loader.pattern_data and 'layout' in self.loader.pattern_data:
            layout = self.loader.pattern_data['layout']
            self.fabric_width = layout.get('width', 1000)
        else:
            self.fabric_width = 1000
        
        logger.info(f"Using fabric width: {self.fabric_width}mm")
        
        # Initialize environment and model
        self.env = None
        self.model = None
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
    
    def setup_environment(self, max_length=None):
        """
        Set up the RL environment.
        
        Args:
            max_length: Maximum fabric length (or None to auto-calculate)
        """
        if not self.pattern_pieces:
            raise ValueError("No pattern pieces loaded")
            
        self.env = FabricEnvironment(
            self.pattern_pieces, 
            self.fabric_width, 
            max_length=max_length,
            grid_size=self.grid_size,
            render_mode="rgb_array"
        )
        
        # Set FreeSewing's layout height if available
        if self.loader.pattern_data and 'layout' in self.loader.pattern_data:
            layout = self.loader.pattern_data['layout']
            if 'height' in layout:
                self.env.set_freesewing_layout_height(layout['height'])
        
        # Wrap environment as per stable_baselines3 requirements
        self.env = DummyVecEnv([lambda: self.env])
        
        logger.info(f"Environment setup with fabric width: {self.fabric_width}mm, "
                    f"grid size: {self.grid_size}mm")
    
    def train(self, total_timesteps=10000, save_path=None):
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Total timesteps for training
            save_path: Path to save the trained model
        """
        if self.env is None:
            self.setup_environment()
            
        # Initialize the model
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading pre-trained model from {self.model_path}")
            self.model = PPO.load(self.model_path, env=self.env)
        else:
            # Determine the device (using MPS if available)
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS device.")
            else:
                device = "cpu"
                logger.info("MPS not available; using CPU.")
            logger.info("Initializing new model")
            self.model = PPO(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                tensorboard_log=self.log_dir,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                device=device  # <-- GPU device set here
            )
        
        # Set up the callback
        class SavingMetricsCallback(BaseCallback):
            def __init__(self, log_dir, check_freq=5000, verbose=1):
                super(SavingMetricsCallback, self).__init__(verbose)
                self.check_freq = check_freq
                self.log_dir = log_dir
                self.metrics = {
                    'timesteps': [],
                    'utilization': [],
                    'fabric_saving': []
                }
                
            def _on_step(self):
                if self.n_calls % self.check_freq == 0:
                    # Use the vectorized environment instead of accessing the raw env
                    vec_env = self.model.get_env()
                    
                    # Reset and run test episode using vectorized env with a maximum step count
                    obs = vec_env.reset()
                    done = [False]
                    test_steps = 0
                    max_test_steps = 500  # limit for debugging
                    while not all(done) and test_steps < max_test_steps:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, _, done, infos = vec_env.step(action)
                        test_steps += 1
                    if test_steps >= max_test_steps:
                        logger.warning("Test episode did not finish within max_test_steps.")
                    
                    # Get info from the last step (for the first environment in the vectorized env)
                    info = infos[0]
                    
                    # Record metrics
                    utilization = info['utilization']
                    fabric_saving = info['fabric_saving']
                    
                    self.metrics['timesteps'].append(self.n_calls)
                    self.metrics['utilization'].append(utilization)
                    self.metrics['fabric_saving'].append(fabric_saving)
                    
                    if self.verbose > 0:
                        logger.info(f"Step {self.n_calls}: Utilization={utilization:.2f}, "
                                    f"Fabric saving={fabric_saving:.2f}")
                    
                    # Save metrics
                    with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
                        json.dump(self.metrics, f)
                    
                    # Optionally save a visualization
                    if self.n_calls % (self.check_freq * 10) == 0:
                        # For visualization, we need to access the unwrapped env
                        env = vec_env.envs[0]
                        env.render(mode='human')
                        # Save as SVG
                        env.export_svg(os.path.join(self.log_dir, f'layout_{self.n_calls}.svg'))
                
                return True
        
        callback = SavingMetricsCallback(log_dir=self.log_dir, check_freq=5000, verbose=1)
        
        # Train the model
        logger.info(f"Training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback, 
                         tb_log_name="freesewing_optimizer")
        
        # Save the trained model
        if save_path:
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
    
    def optimize(self, output_svg=None, render=True):
        """
        Run optimization with the current model and return results.
        
        Args:
            output_svg: Path to save the optimized layout as SVG
            render: Whether to render the layout visualization
            
        Returns:
            Dictionary with optimization results
        """
        if self.env is None:
            self.setup_environment()
            
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                self.model = PPO.load(self.model_path, env=self.env)
            else:
                raise ValueError("No model available. Train a model first or provide a model_path.")
        
        # Get the unwrapped environment
        env = self.env.envs[0]
        
        # Reset the environment
        obs = self.env.reset()
        
        # Loop until all pieces are placed.
        # Added debug prints every 100 iterations to monitor progress.
        done = False
        iteration = 0
        while not done:
            if iteration % 100 == 0:
                logger.info(f"Optimization iteration {iteration} in progress...")
            if iteration >= 15000:
                logger.info("Early stopping at iteration 15000")
                break
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            iteration += 1

        # Ensure info is a dictionary (extract for the first env if necessary)
        if isinstance(info, list):
            info = info[0]

        # Get metrics
        utilization = info['utilization']
        fabric_saving = info['fabric_saving']
        
        # Calculate the used fabric length
        max_y = max([piece.get_transformed_polygon().bounds[3] for piece in self.pattern_pieces])
        
        # Calculate total pattern area
        total_pattern_area = sum(piece.area for piece in self.pattern_pieces)
        
        # Calculate used fabric area
        used_fabric_area = self.fabric_width * max_y
        
        # Calculate waste area
        waste_area = used_fabric_area - total_pattern_area
        
        # Get FreeSewing's layout metrics if available
        freesewing_metrics = {}
        if self.loader.pattern_data and 'layout' in self.loader.pattern_data:
            layout = self.loader.pattern_data['layout']
            if 'width' in layout and 'height' in layout:
                freesewing_area = layout['width'] * layout['height']
                freesewing_metrics = {
                    'width': layout['width'],
                    'height': layout['height'],
                    'area': freesewing_area,
                    'utilization': total_pattern_area / freesewing_area
                }
        
        # Get the layout image
        if render:
            layout_img = env.render()
            plt.figure(figsize=(12, 12))
            plt.imshow(layout_img)
            plt.title('Optimized Fabric Layout')
            plt.show()
        else:
            layout_img = None
        
        # Export to SVG if requested
        svg_metrics = {}
        if output_svg:
            svg_metrics = env.export_svg(output_svg)
        
        # Return results
        results = {
            'utilization': utilization,
            'fabric_saving_percent': fabric_saving * 100,
            'used_fabric_length_mm': max_y,
            'used_fabric_area_mm2': used_fabric_area,
            'total_pattern_area_mm2': total_pattern_area,
            'waste_area_mm2': waste_area,
            'layout_image': layout_img,
            'freesewing_layout': freesewing_metrics,
            'svg_metrics': svg_metrics
        }
        
        # Compare with FreeSewing if metrics available
        if freesewing_metrics:
            results['comparison'] = {
                'area_saving_mm2': freesewing_metrics['area'] - used_fabric_area,
                'area_saving_percent': (freesewing_metrics['area'] - used_fabric_area) / freesewing_metrics['area'] * 100,
                'utilization_improvement': utilization - freesewing_metrics['utilization']
            }
        
        return results


def main():
    """Main function to run the optimizer from command line."""
    parser = argparse.ArgumentParser(description='FreeSewing Pattern Optimizer')
    parser.add_argument('--json', type=str, help='Path to FreeSewing JSON file')
    parser.add_argument('--pdf', type=str, help='Path to PDF file with pattern pieces')
    parser.add_argument('--svg', type=str, help='Path to SVG file with pattern pieces')
    parser.add_argument('--fabric-width', type=int, help='Fabric width in mm')
    parser.add_argument('--grid-size', type=int, default=10, help='Grid size for discretization')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps')
    parser.add_argument('--model-path', type=str, help='Path to load/save model')
    parser.add_argument('--output-svg', type=str, help='Path to save output SVG')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = FreeSewingOptimizer(
        json_path=args.json,
        pdf_path=args.pdf,
        svg_path=args.svg,
        fabric_width=args.fabric_width,
        grid_size=args.grid_size,
        model_path=args.model_path,
        log_dir=args.log_dir
    )
    
    # Train or optimize
    if args.train:
        optimizer.train(total_timesteps=args.timesteps, save_path=args.model_path)
    
    # Always run optimization
    results = optimizer.optimize(output_svg=args.output_svg)
    
    # Print results
    print("\nOptimization Results:")
    print(f"Fabric Utilization: {results['utilization']:.2%}")
    print(f"Fabric Saving: {results['fabric_saving_percent']:.2f}%")
    print(f"Used Fabric Length: {results['used_fabric_length_mm']:.2f} mm")
    print(f"Used Fabric Area: {results['used_fabric_area_mm2']:.2f} mm²")
    print(f"Total Pattern Area: {results['total_pattern_area_mm2']:.2f} mm²")
    print(f"Waste Area: {results['waste_area_mm2']:.2f} mm²")
    
    if 'comparison' in results:
        print("\nComparison with FreeSewing:")
        print(f"Area Saving: {results['comparison']['area_saving_mm2']:.2f} mm² "
              f"({results['comparison']['area_saving_percent']:.2f}%)")
        print(f"Utilization Improvement: {results['comparison']['utilization_improvement']:.2%}")


if __name__ == "__main__":
    main()
