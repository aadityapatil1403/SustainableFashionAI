#!/usr/bin/env python3
"""
Zero Waste Pattern Layout Optimizer - Hybrid Approach

Combines bin packing for initial placement with shape-based refinement
for optimizing pattern piece layouts. Includes enhanced metrics for scrap analysis.
"""

import argparse
import logging
import os
import random
import time
import math
from typing import List, Dict, Tuple, Any, Optional
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from shapely.prepared import prep
from shapely import ops

import numpy as np
import cv2
from pdf2image import convert_from_path
import svgwrite
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A2
from reportlab.lib.units import mm

# Import Shapely for polygon operations
from shapely.geometry import Polygon, Point
import shapely.affinity

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SPACING = 2  # Default spacing between pieces in pixels
DEFAULT_ROTATIONS = [0, 90, 180, 270]  # Default rotation angles to try
DEFAULT_MIN_AREA = 5000  # Minimum contour area to consider
DEFAULT_OPTIMIZATION_ITERATIONS = 2  # Number of optimization passes
MAX_RECTANGLE_COUNT = 10  # Maximum number of rectangles to find in scrap analysis
LARGE_MASK_THRESHOLD = 1000  # Threshold for mask downsampling


class BinPacker:
    """Binary tree bin packing algorithm for initial placement."""
    
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
                    logger.warning(f"Could not place block {block['name']}")
    
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


class LayoutOptimizer:
    """
    Optimizes pattern piece layout using polygon-based refinement.
    Takes a bin-packed layout and refines it for better fabric utilization.
    """
    
    def __init__(self, fabric_width, fabric_height, spacing=DEFAULT_SPACING, 
                 rotations=DEFAULT_ROTATIONS, iterations=DEFAULT_OPTIMIZATION_ITERATIONS):
        self.fabric_width = fabric_width
        self.fabric_height = fabric_height
        self.spacing = spacing
        self.rotations = rotations
        self.iterations = iterations
        self._polygon_cache = {}  # Simple cache for positioned polygons

    def final_overlap_check(self, pattern_pieces):
        """Final sanity check for overlaps with brute force approach."""
        max_attempts = 15  # Increased from 5
        attempt = 0
        fixed_issues = False
        
        while attempt < max_attempts:
            found_issues = False
            
            # First check and fix pieces that are out of bounds
            for piece in pattern_pieces:
                if 'new_x' not in piece or 'new_y' not in piece:
                    continue
                    
                poly = self.get_positioned_polygon(piece)
                minx, miny, maxx, maxy = poly.bounds
                
                # Enforce strict boundary limits with extra margin
                boundary_margin = self.spacing * 6
                
                # Fix out-of-bounds pieces with aggressive repositioning
                if minx < boundary_margin or miny < boundary_margin or maxx > (self.fabric_width - boundary_margin) or maxy > (self.fabric_height - boundary_margin):
                    logger.warning(f"Final boundary check: Piece {piece['name']} is out of bounds: {minx},{miny} - {maxx},{maxy}")
                    
                    # Move piece fully within bounds with extra safety margin
                    if minx < boundary_margin:
                        piece['offset_x'] += (boundary_margin - minx + self.spacing * 2)
                        piece['new_x'] = piece['x'] + piece['offset_x']
                        fixed_issues = True
                        found_issues = True
                        
                    if miny < boundary_margin:
                        piece['offset_y'] += (boundary_margin - miny + self.spacing * 2)
                        piece['new_y'] = piece['y'] + piece['offset_y']
                        fixed_issues = True
                        found_issues = True
                        
                    if maxx > self.fabric_width - boundary_margin:
                        piece['offset_x'] -= (maxx - (self.fabric_width - boundary_margin) + self.spacing * 2)
                        piece['new_x'] = piece['x'] + piece['offset_x']
                        fixed_issues = True
                        found_issues = True
                        
                    if maxy > self.fabric_height - boundary_margin:
                        piece['offset_y'] -= (maxy - (self.fabric_height - boundary_margin) + self.spacing * 2)
                        piece['new_y'] = piece['y'] + piece['offset_y']
                        fixed_issues = True
                        found_issues = True
            
            # Then check for overlaps
            for i, piece1 in enumerate(pattern_pieces):
                if 'new_x' not in piece1 or 'new_y' not in piece1:
                    continue
                    
                poly1 = self.get_positioned_polygon(piece1)
                
                for j, piece2 in enumerate(pattern_pieces):
                    if i == j or 'new_x' not in piece2 or 'new_y' not in piece2:
                        continue
                        
                    poly2 = self.get_positioned_polygon(piece2)
                    
                    # Direct intersection test
                    if poly1.intersects(poly2):
                        logger.warning(f"Final check: Overlap between {piece1['name']} and {piece2['name']}")
                        found_issues = True
                        
                        # Force pieces apart with larger spacing
                        centroid1 = poly1.centroid
                        centroid2 = poly2.centroid
                        dx = centroid1.x - centroid2.x
                        dy = centroid1.y - centroid2.y
                        
                        # Normalize direction
                        dist = max(0.001, math.sqrt(dx*dx + dy*dy))
                        dx /= dist
                        dy /= dist
                        
                        # Use very aggressive separation
                        move_dist = self.spacing * 25  # Increased from 10
                        
                        # Move smaller piece
                        if piece1['area'] < piece2['area']:
                            piece1['offset_x'] += dx * move_dist
                            piece1['offset_y'] += dy * move_dist
                            piece1['new_x'] = piece1['x'] + piece1['offset_x']
                            piece1['new_y'] = piece1['y'] + piece1['offset_y']
                        else:
                            piece2['offset_x'] -= dx * move_dist
                            piece2['offset_y'] -= dy * move_dist
                            piece2['new_x'] = piece2['x'] + piece2['offset_x']
                            piece2['new_y'] = piece2['y'] + piece2['offset_y']
                        
                        fixed_issues = True
                        break
                
                if found_issues:
                    break
            
            # Clear the cache only once per iteration when changes were made
            if found_issues:
                self._polygon_cache = {}
            else:
                break  # No issues found, we're done
                
            attempt += 1
        
        return fixed_issues
        
    def optimize(self, pattern_pieces):
        """
        Optimize the layout of pattern pieces to reduce fabric usage.
        
        Args:
            pattern_pieces: List of pattern pieces with initial positions
                          from bin packing
                          
        Returns:
            Optimized list of pattern pieces
        """
        logger.info(f"Starting layout optimization with {len(pattern_pieces)} pieces")
        
        # Convert to shapely polygons for better handling
        self.prepare_polygons(pattern_pieces)
        
        # Perform multiple optimization iterations
        for iteration in range(self.iterations):
            logger.info(f"Optimization iteration {iteration+1}/{self.iterations}")
            
            # Process each piece
            for i, piece in enumerate(pattern_pieces):
                logger.info(f"Optimizing piece {i+1}/{len(pattern_pieces)}: {piece['name']}")
                
                # Skip pieces without initial position
                if 'new_x' not in piece or 'new_y' not in piece:
                    logger.warning(f"Skipping piece {piece['name']} - no initial position")
                    continue
                
                # Remove this piece from placed pieces for optimization
                other_pieces = [p for p in pattern_pieces if p != piece]
                
                # Try to find a better position for this piece
                success = self.optimize_piece(piece, other_pieces)
                
                if success:
                    logger.info(f"Successfully optimized piece {piece['name']}")
                else:
                    logger.info(f"No better position found for piece {piece['name']}")
            
            # After each iteration, recalculate fabric height
            self.calculate_fabric_dimensions(pattern_pieces)
        
        # Final layout cleanup and dimension calculation
        self.calculate_fabric_dimensions(pattern_pieces)
        
        # Verify no pieces are out of bounds or overlapping
        self.validate_and_fix_layout(pattern_pieces)

        self.final_overlap_check(pattern_pieces)
        
        return pattern_pieces
        
    def prepare_polygons(self, pattern_pieces):
        """
        Prepare pattern pieces for optimization by adding Shapely polygons.
        
        Args:
            pattern_pieces: List of pattern pieces
        """
        for piece in pattern_pieces:
            # Make sure piece has 'new_x' and 'new_y' from bin packing
            if 'new_x' not in piece or 'new_y' not in piece:
                continue
                
            # Create Shapely polygon from points if not already present
            if 'polygon' not in piece:
                try:
                    piece['polygon'] = Polygon(piece['points'])
                    # Fix any potential self-intersections
                    if not piece['polygon'].is_valid:
                        piece['polygon'] = piece['polygon'].buffer(0)
                except Exception as e:
                    logger.error(f"Error creating polygon for {piece['name']}: {e}")
                    # Create a simple convex hull as fallback
                    points = np.array(piece['points'])
                    hull = cv2.convexHull(points.reshape(-1, 1, 2))
                    hull_points = [tuple(p[0]) for p in hull]
                    piece['polygon'] = Polygon(hull_points)
            
            # Add rotation information (initially 0 degrees)
            if 'rotation_angle' not in piece:
                piece['rotation_angle'] = 0
                
            # Calculate position offset from bin packing
            piece['offset_x'] = piece['new_x'] - piece['x']
            piece['offset_y'] = piece['new_y'] - piece['y']
    
    def get_positioned_polygon(self, piece):
        """
        Get the Shapely polygon for a piece in its current position and rotation.
        
        Args:
            piece: The pattern piece
            
        Returns:
            Shapely polygon in correct position
        """
        # Create a cache key using tuple of primitive values
        cache_key = (
            str(piece.get('name', '')),
            float(piece.get('offset_x', 0)),
            float(piece.get('offset_y', 0)),
            float(piece.get('rotation_angle', 0))
        )
        
        # Check if we have this in cache
        if cache_key in self._polygon_cache:
            return self._polygon_cache[cache_key]
        
        # Get the original polygon
        polygon = piece['polygon']
        
        try:
            # Apply rotation if needed
            if piece.get('rotation_angle', 0) != 0:
                # Rotate around centroid
                rotated = shapely.affinity.rotate(polygon, piece['rotation_angle'], origin='centroid')
                
                # Translate to final position
                positioned = shapely.affinity.translate(
                    rotated, 
                    xoff=float(piece['offset_x']),
                    yoff=float(piece['offset_y'])
                )
            else:
                # Just translate if no rotation
                positioned = shapely.affinity.translate(
                    polygon, 
                    xoff=float(piece['offset_x']),
                    yoff=float(piece['offset_y'])
                )
            
            # Fix any invalid geometries
            if not positioned.is_valid:
                positioned = positioned.buffer(0)
            
            # Store in cache    
            self._polygon_cache[cache_key] = positioned
            return positioned
        except Exception as e:
            logger.error(f"Error positioning polygon for {piece.get('name', 'unknown')}: {e}")
            # Return original polygon as fallback
            return polygon
        
    def detect_collision(self, piece, other_pieces, x_offset, y_offset, rotation_angle=0):
        """More robust collision detection with consistent buffering."""
        try:
            # Apply rotation and translation to get the positioned polygon
            polygon = piece['polygon']
            if rotation_angle != 0:
                rotated = shapely.affinity.rotate(polygon, rotation_angle, origin='centroid')
            else:
                rotated = polygon
                
            positioned = shapely.affinity.translate(rotated, xoff=float(x_offset), yoff=float(y_offset))
            
            # Fix any invalid geometries
            if not positioned.is_valid:
                positioned = positioned.buffer(0)
                
            # Check bounds first - use consistent margins
            minx, miny, maxx, maxy = positioned.bounds
            margin = self.spacing * 1.5  # Consistent margin
            
            if (minx < margin or miny < margin or
                maxx > (self.fabric_width - margin) or
                maxy > (self.fabric_height - margin)):
                return True  # Out of bounds
                
            # Use prepared geometry for faster intersection tests
            prepared = prep(positioned.buffer(self.spacing * 1.5))
            
            # Check for collisions
            for other in other_pieces:
                if 'new_x' not in other or 'new_y' not in other:
                    continue
                    
                other_poly = self.get_positioned_polygon(other)
                
                # Use prepared geometry for faster intersection tests
                if prepared.intersects(other_poly):
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"Error in collision detection: {e}")
            return True  # Assume collision on error
            
    def optimize_piece(self, piece, other_pieces):
        """
        Find a better position for a single pattern piece.
        
        Args:
            piece: The piece to optimize
            other_pieces: Other pieces to consider
            
        Returns:
            True if a better position was found, False otherwise
        """
        # Get current position and score
        current_x_offset = piece['offset_x']
        current_y_offset = piece['offset_y']
        current_rotation = piece.get('rotation_angle', 0)
        
        # Evaluate current placement
        current_score = self.evaluate_placement(
            piece, other_pieces, current_x_offset, current_y_offset, current_rotation
        )
        
        best_score = current_score
        best_pos = (current_x_offset, current_y_offset, current_rotation)
        
        # Try different positions around the current one
        positions = self.generate_candidate_positions(piece, other_pieces)
        
        # Evaluate each position
        for pos in positions:
            x_offset, y_offset, rotation = pos
            
            # Skip current position
            if (x_offset == current_x_offset and 
                y_offset == current_y_offset and 
                rotation == current_rotation):
                continue
                
            # Check for collisions
            if self.detect_collision(piece, other_pieces, x_offset, y_offset, rotation):
                continue
                
            # Evaluate placement
            score = self.evaluate_placement(piece, other_pieces, x_offset, y_offset, rotation)
            
            # Update best position if better
            if score > best_score:
                best_score = score
                best_pos = (x_offset, y_offset, rotation)
                
        # If a better position was found, update the piece
        if best_pos != (current_x_offset, current_y_offset, current_rotation):
            x_offset, y_offset, rotation = best_pos
            
            # Store previous values to revert if needed
            previous_offset_x = current_x_offset
            previous_offset_y = current_y_offset
            previous_rotation = current_rotation
            
            # Update piece
            piece['offset_x'] = x_offset
            piece['offset_y'] = y_offset
            piece['rotation_angle'] = rotation
            
            # Update new_x and new_y
            piece['new_x'] = piece['x'] + x_offset
            piece['new_y'] = piece['y'] + y_offset
            
            # Clear polygon cache when piece changes
            self._polygon_cache = {}
            
            # Immediate verification - check if the new position causes issues
            try:
                poly1 = self.get_positioned_polygon(piece)
                
                # Check if piece is out of bounds
                minx, miny, maxx, maxy = poly1.bounds
                if (minx < 0 or miny < 0 or 
                    maxx > self.fabric_width or maxy > self.fabric_height):
                    # Revert to previous position
                    piece['offset_x'] = previous_offset_x
                    piece['offset_y'] = previous_offset_y
                    piece['rotation_angle'] = previous_rotation
                    piece['new_x'] = piece['x'] + previous_offset_x
                    piece['new_y'] = piece['y'] + previous_offset_y
                    self._polygon_cache = {}
                    return False
                
                # Check for overlaps
                for other in other_pieces:
                    if 'new_x' not in other or 'new_y' not in other:
                        continue
                        
                    poly2 = self.get_positioned_polygon(other)
                    if poly1.intersects(poly2):
                        # Revert to previous position
                        piece['offset_x'] = previous_offset_x
                        piece['offset_y'] = previous_offset_y
                        piece['rotation_angle'] = previous_rotation
                        piece['new_x'] = piece['x'] + previous_offset_x
                        piece['new_y'] = piece['y'] + previous_offset_y
                        self._polygon_cache = {}
                        return False
            except Exception as e:
                logger.error(f"Error in verification: {e}")
                # Safety revert
                piece['offset_x'] = previous_offset_x
                piece['offset_y'] = previous_offset_y
                piece['rotation_angle'] = previous_rotation
                piece['new_x'] = piece['x'] + previous_offset_x
                piece['new_y'] = piece['y'] + previous_offset_y
                self._polygon_cache = {}
                return False
                
            return True
        
        return False
        
    def generate_candidate_positions(self, piece, other_pieces):
        """
        Generate candidate positions for optimizing a piece.
        
        Args:
            piece: The piece to optimize
            other_pieces: Other pieces to consider
            
        Returns:
            List of candidate positions (x_offset, y_offset, rotation)
        """
        positions = []
        
        # Get current position
        current_x_offset = piece['offset_x']
        current_y_offset = piece['offset_y']
        
        # 1. Add current position with different rotations
        for rotation in self.rotations:
            positions.append((current_x_offset, current_y_offset, rotation))
            
        # 2. Add positions near edges of other pieces
        for other in other_pieces:
            # Skip pieces without position
            if 'new_x' not in other or 'new_y' not in other:
                continue
                
            try:
                # Create a temporary dict for get_positioned_polygon
                tmp_other = other.copy()
                other_poly = self.get_positioned_polygon(tmp_other)
                bounds = other_poly.bounds
                
                # Try positions around the bounding box of other pieces
                for rotation in self.rotations:
                    # Get rotated piece bounds
                    try:
                        rotated = shapely.affinity.rotate(piece['polygon'], rotation, origin='centroid')
                        rot_width = rotated.bounds[2] - rotated.bounds[0]
                        rot_height = rotated.bounds[3] - rotated.bounds[1]
                    except Exception:
                        # If rotation fails, use original dimensions
                        rot_width = piece['width']
                        rot_height = piece['height']
                    
                    # Add reasonable spacing to avoid immediate collisions
                    safe_spacing = self.spacing * 2
                    
                    # Calculate positions around other piece
                    # Right side of other
                    x_offset = bounds[2] + safe_spacing - piece['x']
                    y_offset = bounds[1] - piece['y']
                    positions.append((x_offset, y_offset, rotation))
                    
                    # Bottom of other
                    x_offset = bounds[0] - piece['x']
                    y_offset = bounds[3] + safe_spacing - piece['y']
                    positions.append((x_offset, y_offset, rotation))
                    
                    # Left side of other (if fits)
                    x_offset = bounds[0] - rot_width - safe_spacing - piece['x']
                    y_offset = bounds[1] - piece['y']
                    positions.append((x_offset, y_offset, rotation))
                    
                    # Top of other (if fits)
                    x_offset = bounds[0] - piece['x']
                    y_offset = bounds[1] - rot_height - safe_spacing - piece['y']
                    positions.append((x_offset, y_offset, rotation))
            except Exception as e:
                logger.warning(f"Error generating positions around piece {other['name']}: {str(e)[:50]}")
                continue
                
        # 3. Add grid-based positions for more comprehensive search
        grid_step = 20  # Step size for grid
        for rotation in self.rotations:
            # Get rotated piece bounds
            try:
                rotated = shapely.affinity.rotate(piece['polygon'], rotation, origin='centroid')
                rot_width = rotated.bounds[2] - rotated.bounds[0]
                rot_height = rotated.bounds[3] - rotated.bounds[1]
            except Exception:
                # If rotation fails, use original dimensions
                rot_width = piece['width']
                rot_height = piece['height']
            
            # Add safety margin to avoid edge cases
            safe_width = rot_width + self.spacing * 2
            safe_height = rot_height + self.spacing * 2
            
            # Generate grid positions
            for x in range(0, int(self.fabric_width - safe_width), grid_step):
                for y in range(0, int(self.fabric_height - safe_height), grid_step):
                    x_offset = x - piece['x']
                    y_offset = y - piece['y']
                    positions.append((x_offset, y_offset, rotation))
                    
        # 4. Limit number of positions to evaluate
        if len(positions) > 500:
            # Prioritize positions: keep all rotations of current position
            current_pos = [(x, y, r) for x, y, r in positions 
                        if x == current_x_offset and y == current_y_offset]
            # And sample from the rest
            other_pos = [(x, y, r) for x, y, r in positions 
                        if x != current_x_offset or y != current_y_offset]
            random.shuffle(other_pos)
            positions = current_pos + other_pos[:500 - len(current_pos)]
        
        # In generate_candidate_positions, add some positions along the edges:
        # Add positions along the left edge
        for y in range(0, int(self.fabric_height - safe_height), grid_step * 2):
            positions.append((- piece['x'], y - piece['y'], rotation))
            
        # Add positions along the top edge
        for x in range(0, int(self.fabric_width - safe_width), grid_step * 2):
            positions.append((x - piece['x'], - piece['y'], rotation))
            
        return positions
        
    def evaluate_placement(self, piece, other_pieces, x_offset, y_offset, rotation):
        """
        Evaluate how good a piece placement is.
        Higher score is better.
        
        Args:
            piece: The piece to evaluate
            other_pieces: Other pieces to consider
            x_offset: X offset for placement
            y_offset: Y offset for placement
            rotation: Rotation angle
            
        Returns:
            Score for this placement
        """
        try:
            # Apply placement to get positioned polygon
            polygon = piece['polygon']
            
            # Apply rotation
            if rotation != 0:
                positioned = shapely.affinity.rotate(polygon, rotation, origin='centroid')
            else:
                positioned = polygon
                
            # Apply translation
            positioned = shapely.affinity.translate(positioned, xoff=x_offset, yoff=y_offset)
            
            # Get bounds
            minx, miny, maxx, maxy = positioned.bounds
            
            # Check if bounds are valid
            if not all(isinstance(v, (int, float)) for v in [minx, miny, maxx, maxy]):
                return float('-inf')  # Invalid bounds, return worst score
                
            # 1. Score for y-position (prefer higher up / smaller y)
            height_score = 100 * (1 - miny / max(self.fabric_height, 1))
            
            # 2. Score for x-position (prefer left / smaller x)
            left_score = 80 * (1 - minx / max(self.fabric_width, 1))
            
            # 3. Score for adjacency to other pieces
            adjacency_score = 0
            min_distance = float('inf')
            
            for other in other_pieces:
                # Skip pieces without position
                if 'new_x' not in other or 'new_y' not in other:
                    continue
                    
                try:
                    # Create a temporary dict for get_positioned_polygon
                    tmp_other = other.copy()
                    other_poly = self.get_positioned_polygon(tmp_other)
                    
                    # Calculate distance
                    try:
                        distance = positioned.distance(other_poly)
                        min_distance = min(min_distance, distance)
                        
                        # Bonus for touching pieces
                        if distance <= self.spacing + 0.1:
                            adjacency_score += 100
                    except Exception:
                        pass
                except Exception:
                    pass
                    
            # Convert min distance to score
            if min_distance != float('inf'):
                adjacency_score += 50 / (min_distance + 1)
            
            # 4. Score for fabric height reduction
            height_reduction_score = 0
            
            # Find max height of placement
            this_max_y = maxy
            
            # Find max height of all other pieces
            other_max_y = 0
            for other in other_pieces:
                if 'new_x' not in other or 'new_y' not in other:
                    continue
                    
                try:
                    # Create a temporary dict for get_positioned_polygon
                    tmp_other = other.copy()
                    other_poly = self.get_positioned_polygon(tmp_other)
                    _, _, _, other_maxy = other_poly.bounds
                    other_max_y = max(other_max_y, other_maxy)
                except Exception:
                    pass
            
            # Check if this placement increases fabric height
            if this_max_y <= other_max_y:
                # Doesn't increase height, big bonus
                height_reduction_score = 300
            else:
                # Penalize height increase
                height_increase = this_max_y - other_max_y
                height_reduction_score = 100 / (height_increase + 1)
            
            # 5. Bonus for standard rotations
            rotation_score = 0
            if rotation in [0, 90, 180, 270]:
                rotation_score = 30
                
            # 6. Final score
            total_score = (height_score + left_score + adjacency_score + 
                        height_reduction_score + rotation_score)
                        
            return total_score
        except Exception as e:
            logger.error(f"Error evaluating placement: {e}")
            return float('-inf')  # Return worst score on error
        
    def calculate_fabric_dimensions(self, pattern_pieces):
        """
        Calculate the actual fabric dimensions needed for the current layout
        but respect the fixed dimensions provided at initialization.
        
        Args:
            pattern_pieces: List of pattern pieces
            
        Returns:
            Tuple of (width, height)
        """
        max_x = 0
        max_y = 0
        
        for piece in pattern_pieces:
            # Skip pieces without position
            if 'new_x' not in piece or 'new_y' not in piece:
                continue
                
            # Get positioned polygon
            try:
                # Create a temporary piece to avoid modifying the original
                temp_piece = {
                    'name': piece.get('name', ''),
                    'polygon': piece.get('polygon'),
                    'offset_x': float(piece.get('offset_x', 0)),
                    'offset_y': float(piece.get('offset_y', 0)),
                    'rotation_angle': float(piece.get('rotation_angle', 0))
                }
                
                poly = self.get_positioned_polygon(temp_piece)
                
                # Update max dimensions
                bounds = poly.bounds
                if len(bounds) == 4:
                    _, _, piece_max_x, piece_max_y = bounds
                    max_x = max(max_x, piece_max_x)
                    max_y = max(max_y, piece_max_y)
            except Exception as e:
                logger.warning(f"Error calculating bounds for piece {piece.get('name', 'unknown')}: {e}")
                # Fallback to bounding box if Shapely operations fail
                if 'width' in piece and 'height' in piece:
                    piece_max_x = float(piece['new_x']) + float(piece['width'])
                    piece_max_y = float(piece['new_y']) + float(piece['height'])
                    max_x = max(max_x, piece_max_x)
                    max_y = max(max_y, piece_max_y)
        
        # Add margin
        max_x += self.spacing
        max_y += self.spacing
        
        # Here's the key change: respect the original fabric dimensions
        # Update width only if necessary, but keep height fixed
        self.fabric_width = max(self.fabric_width, max_x)
        # DO NOT update the height - keep it fixed
        
        return self.fabric_width, self.fabric_height
    
    def validate_and_fix_layout(self, pattern_pieces):
        """
        Validate the final layout and fix any out-of-bounds or overlapping pieces
        with a more aggressive approach to ensure no overlaps remain.
        """
        made_changes = False
        max_fix_attempts = 10  # Increased from 5 to allow more fix attempts
        attempt = 0
        
        # Increase spacing for collision detection
        collision_spacing = self.spacing * 3  # Triple the spacing for more aggressive separation
        
        while attempt < max_fix_attempts:
            has_issues = False
            
            # First fix out-of-bounds pieces
            for piece in pattern_pieces:
                if 'new_x' not in piece or 'new_y' not in piece:
                    continue
                    
                try:
                    # Check if piece is valid
                    poly = self.get_positioned_polygon(piece)
                    minx, miny, maxx, maxy = poly.bounds
                    
                    # Fix out-of-bounds pieces with larger margins
                    margin = collision_spacing
                    if minx < margin or miny < margin or maxx > (self.fabric_width - margin) or maxy > (self.fabric_height - margin):
                        logger.warning(f"Piece {piece['name']} is out of bounds: {minx},{miny} - {maxx},{maxy}")
                        
                        # Move piece within bounds with larger safety margins
                        if minx < margin:
                            piece['offset_x'] += (margin - minx + collision_spacing)  # Extra spacing
                            piece['new_x'] = piece['x'] + piece['offset_x']
                            made_changes = True
                            has_issues = True
                            
                        if miny < margin:
                            piece['offset_y'] += (margin - miny + collision_spacing)  # Extra spacing
                            piece['new_y'] = piece['y'] + piece['offset_y']
                            made_changes = True
                            has_issues = True
                            
                        if maxx > self.fabric_width - margin:
                            piece['offset_x'] -= (maxx - (self.fabric_width - margin) + collision_spacing)  # Extra spacing
                            piece['new_x'] = piece['x'] + piece['offset_x']
                            made_changes = True
                            has_issues = True
                            
                        if maxy > self.fabric_height - margin:
                            piece['offset_y'] -= (maxy - (self.fabric_height - margin) + collision_spacing)  # Extra spacing
                            piece['new_y'] = piece['y'] + piece['offset_y']
                            made_changes = True
                            has_issues = True
                except Exception as e:
                    logger.error(f"Error validating piece {piece.get('name', 'unknown')}: {e}")
            
            # Then check for overlaps with much more aggressive fixing
            overlaps_detected = False
            
            # First, build a grid system to quickly identify potential collisions
            # This reduces the O(n²) problem to near-linear time
            grid_size = 50  # Size of grid cells
            collision_grid = {}
            
            # Add pieces to grid
            for i, piece in enumerate(pattern_pieces):
                if 'new_x' not in piece or 'new_y' not in piece:
                    continue
                    
                try:
                    poly = self.get_positioned_polygon(piece)
                    minx, miny, maxx, maxy = poly.bounds
                    
                    # Get grid cells that this piece intersects
                    grid_cells = set()
                    for x in range(int(minx) // grid_size, int(maxx) // grid_size + 1):
                        for y in range(int(miny) // grid_size, int(maxy) // grid_size + 1):
                            grid_cells.add((x, y))
                            if (x, y) not in collision_grid:
                                collision_grid[(x, y)] = []
                            collision_grid[(x, y)].append(i)
                            
                    # Store grid cells with piece for later use
                    piece['grid_cells'] = grid_cells
                except Exception as e:
                    logger.error(f"Error building collision grid for {piece.get('name', 'unknown')}: {e}")
            
            # Now check for overlaps using the grid
            for i, piece1 in enumerate(pattern_pieces):
                if 'new_x' not in piece1 or 'new_y' not in piece1 or 'grid_cells' not in piece1:
                    continue
                    
                # Get potential collision candidates from grid
                candidates = set()
                for cell in piece1['grid_cells']:
                    if cell in collision_grid:
                        candidates.update(collision_grid[cell])
                
                # Remove self from candidates
                candidates.discard(i)
                
                # Check actual collisions with candidates only
                for j in candidates:
                    piece2 = pattern_pieces[j]
                    if 'new_x' not in piece2 or 'new_y' not in piece2:
                        continue
                    
                    try:
                        poly1 = self.get_positioned_polygon(piece1)
                        poly2 = self.get_positioned_polygon(piece2)
                        
                        # Use a much more aggressive buffering for detecting overlaps
                        buffer_distance = collision_spacing
                        buffered_poly1 = poly1.buffer(buffer_distance)
                        buffered_poly2 = poly2.buffer(buffer_distance)
                        
                        if buffered_poly1.intersects(buffered_poly2):
                            logger.warning(f"Overlap detected between pieces {piece1['name']} and {piece2['name']}")
                            overlaps_detected = True
                            
                            # Determine which piece to move based on size and position
                            if piece1['area'] < piece2['area']:
                                smaller, larger = piece1, piece2
                            else:
                                smaller, larger = piece2, piece1
                                
                            # Find direction to move (away from other piece's centroid)
                            centroid1 = poly1.centroid
                            centroid2 = poly2.centroid
                            dx = centroid1.x - centroid2.x
                            dy = centroid1.y - centroid2.y
                            
                            # Normalize direction
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist < 0.001:  # Avoid division by zero
                                dx, dy = 1.0, 0.0  # Default direction if centroids overlap
                            else:
                                dx /= dist
                                dy /= dist
                                
                            # Move smaller piece away by a much larger distance
                            move_dist = collision_spacing * 10  # Very aggressive movement to ensure separation
                            smaller['offset_x'] += dx * move_dist
                            smaller['offset_y'] += dy * move_dist
                            smaller['new_x'] = smaller['x'] + smaller['offset_x']
                            smaller['new_y'] = smaller['y'] + smaller['offset_y']
                            made_changes = True
                            has_issues = True
                            
                            # Break after first overlap for this piece to rebuild grid
                            break
                    except Exception as e:
                        logger.error(f"Error checking overlap: {str(e)[:100]}")
            
            # Clear grid cell references after each iteration
            for piece in pattern_pieces:
                if 'grid_cells' in piece:
                    del piece['grid_cells']
            
            # If no issues were found in this iteration, we're done
            if not has_issues:
                break

            # Double-check every piece against every other piece directly (no grid)
            for i, piece1 in enumerate(pattern_pieces):
                if 'new_x' not in piece1 or 'new_y' not in piece1:
                    continue
                    
                poly1 = self.get_positioned_polygon(piece1)
                
                for j, piece2 in enumerate(pattern_pieces[i+1:], i+1):
                    if 'new_x' not in piece2 or 'new_y' not in piece2:
                        continue
                        
                    poly2 = self.get_positioned_polygon(piece2)
                    
                    if poly1.intersects(poly2):
                        logger.warning(f"Direct check: Overlap between {piece1['name']} and {piece2['name']}")
                        overlaps_detected = True
                        # Move both pieces apart slightly
                        vector = shapely.ops.nearest_points(poly1, poly2)
                        dx = vector[1].x - vector[0].x
                        dy = vector[1].y - vector[0].y
                        
                        # Normalize and apply minimum movement
                        dist = max(0.001, math.sqrt(dx*dx + dy*dy))
                        dx /= dist
                        dy /= dist
                        
                        # Move pieces in opposite directions
                        move_dist = self.spacing * 5
                        piece1['offset_x'] -= dx * move_dist/2
                        piece1['offset_y'] -= dy * move_dist/2
                        piece1['new_x'] = piece1['x'] + piece1['offset_x']
                        piece1['new_y'] = piece1['y'] + piece1['offset_y']
                        
                        piece2['offset_x'] += dx * move_dist/2
                        piece2['offset_y'] += dy * move_dist/2
                        piece2['new_x'] = piece2['x'] + piece2['offset_x']
                        piece2['new_y'] = piece2['y'] + piece2['offset_y']
                        
                        made_changes = True
                        has_issues = True 

            # Clear cache and recalculate dimensions for the next iteration
            self._polygon_cache = {}
            self.calculate_fabric_dimensions(pattern_pieces)
            attempt += 1
        
        # Do a final aggressive spacing pass if we detected overlaps
        if overlaps_detected and attempt >= max_fix_attempts:
            logger.warning("Maximum fix attempts reached with overlaps still present. Performing emergency spacing.")
            
            # Sort pieces by area (largest first)
            sorted_pieces = sorted(pattern_pieces, key=lambda p: p.get('area', 0) if 'area' in p else 0, reverse=True)
            
            # Position pieces one-by-one with large gaps
            current_x, current_y = collision_spacing, collision_spacing
            row_height = 0
            
            for piece in sorted_pieces:
                if 'width' not in piece or 'height' not in piece:
                    continue
                    
                piece_width = piece['width']
                piece_height = piece['height']
                
                # Check if piece fits in current row
                if current_x + piece_width + collision_spacing > self.fabric_width:
                    # Move to next row
                    current_x = collision_spacing
                    current_y += row_height + collision_spacing
                    row_height = 0
                
                # Set new position
                piece['offset_x'] = current_x - piece['x']
                piece['offset_y'] = current_y - piece['y']
                piece['new_x'] = current_x
                piece['new_y'] = current_y
                
                # Update position and row height
                current_x += piece_width + collision_spacing * 2
                row_height = max(row_height, piece_height)
                
                # Reset rotation to reduce complexity
                piece['rotation_angle'] = 0
                
                made_changes = True
            
            # Clear cache and recalculate dimensions
            self._polygon_cache = {}
            self.calculate_fabric_dimensions(pattern_pieces)
        
        # Final check to absolutely ensure no pieces are out of bounds
        for piece in pattern_pieces:
            if 'new_x' not in piece or 'new_y' not in piece:
                continue
                
            try:
                poly = self.get_positioned_polygon(piece)
                minx, miny, maxx, maxy = poly.bounds
                
                # Just ensure absolutely no pieces go outside fabric boundary
                if minx < 0:
                    piece['offset_x'] += abs(minx) + self.spacing
                    piece['new_x'] = piece['x'] + piece['offset_x']
                    made_changes = True
                    
                if miny < 0:
                    piece['offset_y'] += abs(miny) + self.spacing
                    piece['new_y'] = piece['y'] + piece['offset_y']
                    made_changes = True
                    
                if maxx > self.fabric_width:
                    piece['offset_x'] -= (maxx - self.fabric_width) + self.spacing
                    piece['new_x'] = piece['x'] + piece['offset_x']
                    made_changes = True
                    
                if maxy > self.fabric_height:
                    piece['offset_y'] -= (maxy - self.fabric_height) + self.spacing
                    piece['new_y'] = piece['y'] + piece['offset_y']
                    made_changes = True
            except Exception:
                pass
        
        if made_changes:
            logger.info("Fixed layout issues. Recalculating dimensions...")
            self.calculate_fabric_dimensions(pattern_pieces)
            
        return not made_changes


# New class for scrap analysis
class ScrapAnalyzer:
    """Simplified analyzer for fabric scrap focusing on key metrics."""
    
    def __init__(self, fabric_width, fabric_height):
        self.fabric_width = int(fabric_width)
        self.fabric_height = int(fabric_height)
        
    def analyze(self, pattern_pieces):
        """
        Analyze scrap in a pattern layout focusing on usable fabric metrics.
        
        Args:
            pattern_pieces: List of pattern pieces with positions
            
        Returns:
            Dictionary of scrap metrics
        """
        logger.info("Analyzing fabric scrap...")
        
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
                # Get adjusted points considering rotation and position
                points = piece['points']
                dx = float(piece['new_x'] - piece['x'])
                dy = float(piece['new_y'] - piece['y'])
                
                if 'rotation_angle' in piece and piece['rotation_angle'] != 0:
                    # Calculate centroid
                    centroid_x = sum(p[0] for p in points) / len(points)
                    centroid_y = sum(p[1] for p in points) / len(points)
                    angle_rad = math.radians(piece['rotation_angle'])
                    
                    # Apply rotation around centroid
                    adjusted_points = []
                    for px, py in points:
                        # Translate to origin
                        px_centered = px - centroid_x
                        py_centered = py - centroid_y
                        
                        # Rotate
                        px_rotated = (px_centered * math.cos(angle_rad) - 
                                    py_centered * math.sin(angle_rad))
                        py_rotated = (px_centered * math.sin(angle_rad) + 
                                    py_centered * math.cos(angle_rad))
                        
                        # Translate back and apply offset
                        adjusted_points.append((
                            int(px_rotated + centroid_x + dx),
                            int(py_rotated + centroid_y + dy)
                        ))
                else:
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
                logger.warning(f"Error adding piece {piece.get('name', 'unknown')} to mask: {e}")
        
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
                # Return safe values if out of bounds
                logger.warning("Rectangle out of bounds after scaling, using safe values")
                max_rect = (0, 0, 0, 0)
                max_area = 0
        
        # Final bounds check
        if max_rect[0] < 0 or max_rect[1] < 0 or max_rect[2] <= 0 or max_rect[3] <= 0:
            return 0, (0, 0, 0, 0)
        
        return max_area, max_rect
    
    # In the ScrapAnalyzer class, modify the _find_multiple_rectangles method:

    # In the ScrapAnalyzer class, completely rewrite the _find_multiple_rectangles method:
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
    
    def _process_rectangle(self, params):
        """Process a single rectangle search - for parallel processing."""
        empty_mask, min_area = params
        rect_area, rect_coords = self._find_largest_rectangle(empty_mask)
        if rect_area >= min_area:
            return {
                'area': rect_area,
                'coords': rect_coords
            }
        return None

    

def extract_pattern_pieces(pdf_path, min_area=DEFAULT_MIN_AREA):
    """Extract pattern pieces from a PDF file with enhanced processing."""
    logger.info(f"Converting PDF to image: {pdf_path}")
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=150)
    
    # Process the first page
    img = np.array(images[0])
    # Convert RGB to BGR (for OpenCV)
    img = img[:, :, ::-1].copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold with adaptive method for better extraction
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Optional: Use morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pattern_pieces = []
    for i, contour in enumerate(contours):
        # Filter small contours (noise)
        contour_area = cv2.contourArea(contour)
        if contour_area < min_area:
            continue
        
        # Simplify contour (reduce points but maintain shape)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to simple polygon
        points = [(int(point[0][0]), int(point[0][1])) for point in approx]
        
        # Skip if too few points
        if len(points) < 3:
            continue
            
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
            'area': Polygon(points).area,
            'color_index': i  # Used for consistent coloring between layouts
        }
        
        pattern_pieces.append(piece)
    
    logger.info(f"Extracted {len(pattern_pieces)} pattern pieces")
    return pattern_pieces, img.shape[1], img.shape[0]

def create_svg(pattern_pieces, fabric_width, fabric_height, scrap_analysis=None, output_path="optimized_layout.svg"):
    """Create SVG visualization of the pattern layout with scrap analysis highlights."""
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
        
        # Get points
        points = piece['points']
        
        # Check if we need to rotate
        if 'rotation_angle' in piece and piece['rotation_angle'] != 0:
            # Calculate centroid for rotation
            centroid_x = sum(p[0] for p in points) / len(points)
            centroid_y = sum(p[1] for p in points) / len(points)
            
            # Rotate points around centroid
            angle_rad = math.radians(piece['rotation_angle'])
            rotated_points = []
            
            for px, py in points:
                # Translate to origin
                px_centered = px - centroid_x
                py_centered = py - centroid_y
                
                # Rotate
                px_rotated = (px_centered * math.cos(angle_rad) - 
                             py_centered * math.sin(angle_rad))
                py_rotated = (px_centered * math.sin(angle_rad) + 
                             py_centered * math.cos(angle_rad))
                
                # Translate back
                rotated_points.append((
                    px_rotated + centroid_x + dx,
                    py_rotated + centroid_y + dy
                ))
                
            adjusted_points = rotated_points
        else:
            # Just translate
            adjusted_points = [(p[0] + dx, p[1] + dy) for p in points]
        
        # Add polygon
        dwg.add(dwg.polygon(adjusted_points, fill=color, stroke='black', stroke_width=1))
        
        # Add label with piece name and rotation
        rotation_text = f" (R{piece['rotation_angle']}°)" if piece.get('rotation_angle', 0) != 0 else ""
        
        # Calculate centroid for text placement
        x_coords = [p[0] for p in adjusted_points]
        y_coords = [p[1] for p in adjusted_points]
        centroid_x = sum(x_coords) / len(adjusted_points)
        centroid_y = sum(y_coords) / len(adjusted_points)
        
        dwg.add(dwg.text(
            piece['name'] + rotation_text, 
            insert=(centroid_x, centroid_y), 
            text_anchor='middle', 
            font_size='12'
        ))
    
    # Add scrap analysis legend if available
    if scrap_analysis:
        y_pos = 20
        dwg.add(dwg.text(
            "Scrap Analysis", 
            insert=(fabric_width - 200, y_pos), 
            font_size='14', font_weight='bold'
        ))
        y_pos += 15
        
        # Calculate and add scrap ratio
        total_area = fabric_width * fabric_height
        scrap_ratio = scrap_analysis['total_scrap_area'] / total_area if total_area > 0 else 0
        dwg.add(dwg.text(
            f"Scrap ratio: {scrap_ratio*100:.1f}%", 
            insert=(fabric_width - 200, y_pos), 
            font_size='10'
        ))
        y_pos += 12
        
        # Add scrap quality index
        if 'scrap_quality_index' in scrap_analysis:
            dwg.add(dwg.text(
                f"Scrap quality index: {scrap_analysis['scrap_quality_index']:.1f}%", 
                insert=(fabric_width - 200, y_pos), 
                font_size='10'
            ))
            y_pos += 12
        
        # Add usable rectangles count
        if 'usable_rectangles' in scrap_analysis:
            rect_count = len(scrap_analysis['usable_rectangles'])
            dwg.add(dwg.text(
                f"Usable rectangles: {rect_count}", 
                insert=(fabric_width - 200, y_pos), 
                font_size='10'
            ))
            y_pos += 12
        
        # Add scrap regions count
        dwg.add(dwg.text(
            f"Scrap regions: {scrap_analysis['scrap_regions']}", 
            insert=(fabric_width - 200, y_pos), 
            font_size='10'
        ))
            
    # Save drawing
    dwg.save()
    logger.info(f"SVG saved to {output_path}")
    
    return output_path


# Fix the rectangle drawing bounds check in create_pdf function
# Original layout section corrected to check against orig_width and orig_height instead of fabric_width

def create_pdf(pattern_pieces, original_pieces, original_width, original_height, 
               fabric_width, fabric_height, stats, 
               original_analysis, optimized_analysis, output_pdf="optimized_layout.pdf"):
    """Create PDF with original and optimized layouts plus comparison statistics."""
    # Use larger page size
    page_width, page_height = A2
    c = canvas.Canvas(output_pdf, pagesize=(page_width, page_height))
    
    # Calculate margins and layout positioning
    margin = 20 * mm
    available_height = page_height - 2 * margin
    available_width = page_width - 2 * margin
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, page_height - margin, "Hybrid (Bin Packing + NFP) Pattern Piece Optimization")
    
    # Statistics section
    c.setFont("Helvetica", 10)
    y_pos = page_height - margin - 20 * mm
    
    # Draw comparison statistics
    section = None
    in_section = False  # Track if we're currently inside a section with saved state

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
        y_pos -= 4.5 * mm
    
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
    for i, piece in enumerate(original_pieces):
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
    
    # Draw original multiple rectangles if available - FIXED BOUNDS CHECK
    if original_analysis and 'usable_rectangles' in original_analysis:
        # Create a list of filtered rectangles that are definitely within bounds
        valid_rectangles = []
        for rect_data in original_analysis['usable_rectangles']:
            rect = rect_data['coords']
            x, y, w, h = rect
            
            # Super strict validation - only accept completely valid rectangles
            # FIXED: Check against orig_width and orig_height, not fabric dimensions
            if (isinstance(x, (int, float)) and isinstance(y, (int, float)) and 
                isinstance(w, (int, float)) and isinstance(h, (int, float)) and
                x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= orig_width and y + h <= orig_height):
                valid_rectangles.append(rect_data)
        
        # Now draw only the validated rectangles
        for i, rect_data in enumerate(valid_rectangles):
            rect = rect_data['coords']
            x, y, w, h = rect
            
            # Extra safety check - FIXED: using orig_width and orig_height
            if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= orig_width and y + h <= orig_height):
                # Vary shade of green for different rectangles
                green_level = max(0.3, 0.8 - i * 0.1)
                c.setStrokeColorRGB(0, 0.8, 0)
                c.setFillColorRGB(0, green_level, 0, 0.2)
                c.setDash(5, 5)
                c.rect(x, y, w, h, fill=1, stroke=1)
    elif original_analysis and 'largest_rectangle_coords' in original_analysis:
        rect = original_analysis['largest_rectangle_coords']
        x, y, w, h = rect
        
        # Apply same strict validation - FIXED: using orig_width and orig_height
        if (isinstance(x, (int, float)) and isinstance(y, (int, float)) and 
            isinstance(w, (int, float)) and isinstance(h, (int, float)) and
            x >= 0 and y >= 0 and w > 0 and h > 0 and 
            x + w <= orig_width and y + h <= orig_height):
            c.setStrokeColorRGB(0, 0.8, 0)
            c.setFillColorRGB(0, 0.8, 0, 0.2)
            c.setDash(5, 5)
            c.rect(x, y, w, h, fill=1, stroke=1)
    
    # Important: restore state before moving to next section
    c.restoreState()
    
    # Rest of the function remains unchanged
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
        
        # Get points with rotation and translation applied
        points = piece['points']
        
        # Calculate position offset to move points relative to new position
        dx = piece['new_x'] - piece['x']
        dy = piece['new_y'] - piece['y']
        
        # Check if we need to rotate
        if 'rotation_angle' in piece and piece['rotation_angle'] != 0:
            # Calculate centroid for rotation
            centroid_x = sum(p[0] for p in points) / len(points)
            centroid_y = sum(p[1] for p in points) / len(points)
            
            # Rotate points around centroid
            angle_rad = math.radians(piece['rotation_angle'])
            adjusted_points = []
            
            for px, py in points:
                # Translate to origin
                px_centered = px - centroid_x
                py_centered = py - centroid_y
                
                # Rotate
                px_rotated = (px_centered * math.cos(angle_rad) - 
                             py_centered * math.sin(angle_rad))
                py_rotated = (px_centered * math.sin(angle_rad) + 
                             py_centered * math.cos(angle_rad))
                
                # Translate back
                adjusted_points.append((
                    px_rotated + centroid_x + dx,
                    py_rotated + centroid_y + dy
                ))
        else:
            # Just translate
            adjusted_points = [(p[0] + dx, p[1] + dy) for p in points]
        
        # Draw polygon
        path = c.beginPath()
        path.moveTo(adjusted_points[0][0], adjusted_points[0][1])
        for point in adjusted_points[1:]:
            path.lineTo(point[0], point[1])
        path.close()
        c.drawPath(path, fill=1, stroke=1)
        
        # Add rotation indicator
        if 'rotation_angle' in piece and piece['rotation_angle'] != 0:
            c.setFont("Helvetica", 8)
            c.setFillColorRGB(0, 0, 0)
            
            # Calculate centroid
            centroid_x = sum(p[0] for p in adjusted_points) / len(adjusted_points)
            centroid_y = sum(p[1] for p in adjusted_points) / len(adjusted_points)
            
            c.drawCentredString(
                centroid_x, centroid_y, 
                f"(R{piece['rotation_angle']}°)"
            )
    
    # Draw optimized usable rectangles if available
    if optimized_analysis and 'usable_rectangles' in optimized_analysis:
        # Create a list of filtered rectangles that are definitely within bounds
        valid_rectangles = []
        for rect_data in optimized_analysis['usable_rectangles']:
            rect = rect_data['coords']
            x, y, w, h = rect
            
            # Super strict validation - only accept completely valid rectangles
            if (isinstance(x, (int, float)) and isinstance(y, (int, float)) and 
                isinstance(w, (int, float)) and isinstance(h, (int, float)) and
                x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= fabric_width and y + h <= fabric_height):
                valid_rectangles.append(rect_data)
        
        # Now draw only the validated rectangles
        for i, rect_data in enumerate(valid_rectangles):
            rect = rect_data['coords']
            x, y, w, h = rect
            
            # Extra safety check
            if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= fabric_width and y + h <= fabric_height):
                # Vary shade of green for different rectangles
                green_level = max(0.3, 0.8 - i * 0.1)
                c.setStrokeColorRGB(0, 0.8, 0)
                c.setFillColorRGB(0, green_level, 0, 0.2)
                c.setDash(5, 5)
                c.rect(x, y, w, h, fill=1, stroke=1)
    elif optimized_analysis and 'largest_rectangle_coords' in optimized_analysis:
        rect = optimized_analysis['largest_rectangle_coords']
        x, y, w, h = rect
        
        # Apply same strict validation
        if (isinstance(x, (int, float)) and isinstance(y, (int, float)) and 
            isinstance(w, (int, float)) and isinstance(h, (int, float)) and
            x >= 0 and y >= 0 and w > 0 and h > 0 and 
            x + w <= orig_width and y + h <= orig_height):
            c.setStrokeColorRGB(0, 0.8, 0)
            c.setFillColorRGB(0, 0.8, 0, 0.2)
            c.setDash(5, 5)
            c.rect(x, y, w, h, fill=1, stroke=1)
    
    # Ensure we restore the state before continuing
    c.restoreState()
    
    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(margin, margin/2, "Green highlighted rectangles represent reusable fabric pieces")
    
    # Save the PDF
    c.save()
    logger.info(f"PDF saved to {output_pdf}")
    
    return output_pdf

def calculate_original_fabric_area(pattern_pieces):
    """Calculate the minimum rectangle that contains all pattern pieces in original layout."""
    if not pattern_pieces:
        return 0, 0, 0, 0, 0
    
    # Calculate bounding box of all pieces - add a small buffer
    buffer = 10  # Small buffer to avoid precise edge issues
    min_x = max(0, min(min(p[0] for p in piece['points']) for piece in pattern_pieces) - buffer)
    min_y = max(0, min(min(p[1] for p in piece['points']) for piece in pattern_pieces) - buffer)
    max_x = max(max(p[0] for p in piece['points']) for piece in pattern_pieces) + buffer
    max_y = max(max(p[1] for p in piece['points']) for piece in pattern_pieces) + buffer
    
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
    
    return min_x, min_y, width, height, area

def apply_seam_allowance(pattern_pieces, seam_allowance=10):
    """
    Apply seam allowance to pattern pieces by expanding their polygons.
    
    Args:
        pattern_pieces: List of pattern pieces
        seam_allowance: Seam allowance in pixels (default: 10)
        
    Returns:
        Pattern pieces with expanded polygons
    """
    if seam_allowance <= 0:
        return pattern_pieces
        
    logger.info(f"Applying seam allowance of {seam_allowance} pixels to pattern pieces")
    
    for piece in pattern_pieces:
        try:
            # Skip pieces without polygons
            if 'polygon' not in piece:
                continue
                
            # Get original polygon
            original_polygon = piece['polygon']
            
            # Skip invalid polygons
            if not original_polygon.is_valid:
                logger.warning(f"Skipping invalid polygon in piece {piece.get('name', 'unknown')}")
                continue
                
            # Apply buffer to expand the polygon by seam allowance
            expanded_polygon = original_polygon.buffer(seam_allowance)
            
            # Skip if buffer failed
            if not expanded_polygon.is_valid:
                logger.warning(f"Failed to add seam allowance to piece {piece.get('name', 'unknown')}")
                continue
                
            # Update polygon in piece
            piece['polygon'] = expanded_polygon
            
            # Update points for drawing
            if isinstance(expanded_polygon, Polygon):
                # Simple polygon case
                piece['points'] = list(expanded_polygon.exterior.coords)[:-1]  # Exclude last point (duplicate of first)
            else:
                # MultiPolygon case - use largest polygon
                largest_polygon = max(expanded_polygon, key=lambda p: p.area)
                piece['points'] = list(largest_polygon.exterior.coords)[:-1]
            
            # Update dimensions
            minx, miny, maxx, maxy = expanded_polygon.bounds
            piece['width'] = maxx - minx
            piece['height'] = maxy - miny
            
            # Update area
            piece['area'] = expanded_polygon.area
            
        except Exception as e:
            logger.error(f"Error adding seam allowance to piece {piece.get('name', 'unknown')}: {e}")
    
    return pattern_pieces

def optimize_pattern_layout(input_pdf, output_prefix, fabric_width=None, spacing=DEFAULT_SPACING,
                           rotations=DEFAULT_ROTATIONS, optimization_iterations=DEFAULT_OPTIMIZATION_ITERATIONS,
                           min_area=DEFAULT_MIN_AREA, seam_allowance=10):
    """Main function to optimize pattern layout using hybrid approach with scrap analysis."""
    start_time = time.time()
    
    # Step 1: Extract pattern pieces from PDF
    pattern_pieces, original_width, original_height = extract_pattern_pieces(input_pdf, min_area)
    
    if not pattern_pieces:
        logger.error("No pattern pieces were extracted.")
        return
    
    # Calculate original layout area
    orig_min_x, orig_min_y, orig_width, orig_height, original_fabric_area = calculate_original_fabric_area(pattern_pieces)
    
    # Create a copy of pattern pieces in original positions (before seam allowance)
    original_pieces = []
    for piece in pattern_pieces:
        original_piece = piece.copy()
        original_piece['new_x'] = original_piece['x']
        original_piece['new_y'] = original_piece['y']
        original_piece['offset_x'] = 0
        original_piece['offset_y'] = 0
        original_piece['rotation_angle'] = 0
        original_pieces.append(original_piece)
    
    # Step 2: Analyze the ORIGINAL layout for baseline metrics
    logger.info("Analyzing original layout...")
    original_analyzer = ScrapAnalyzer(orig_width, orig_height)
    original_analysis = original_analyzer.analyze(original_pieces)
    
    # Step 2.5: Apply seam allowance to pattern pieces
    if seam_allowance > 0:
        logger.info(f"Applying seam allowance of {seam_allowance} pixels")
        # Add seam allowance to all pattern pieces
        apply_seam_allowance(pattern_pieces, seam_allowance)
        
        # Recalculate dimensions after adding seam allowance
        _, _, orig_width_with_seam, orig_height_with_seam, original_fabric_area_with_seam = calculate_original_fabric_area(pattern_pieces)
        logger.info(f"After seam allowance: width={orig_width_with_seam}, height={orig_height_with_seam} (was: width={orig_width}, height={orig_height})")
    
    # Use a reasonable fabric width if not provided
    # But keep the original dimensions for comparison
    if fabric_width is None:
        # Use original width or 90% of it if it's very large
        fabric_width = min(orig_width, int(original_width * 0.9))
    
    # Set fabric height to original height - this is the key change
    fabric_height = orig_height
    
    logger.info(f"Using fixed fabric dimensions: {fabric_width}x{fabric_height}")
    
    # Calculate total pattern area
    total_pattern_area = sum(piece['area'] for piece in pattern_pieces)
    
    # Step 3: Initial placement with bin packing
    logger.info("Performing initial bin packing placement")
    bin_packer = BinPacker(fabric_width)
    bin_packer.fit(pattern_pieces)
    
    # Check how many pieces were placed
    placed_count = sum(1 for piece in pattern_pieces if 'new_x' in piece)
    logger.info(f"Bin packing placed {placed_count}/{len(pattern_pieces)} pieces")
    
    # Step 4: Optimize with shape-based refinement but maintain fixed dimensions
    logger.info("Performing polygon-based layout refinement within fixed dimensions")
    optimizer = LayoutOptimizer(
        fabric_width, fabric_height,  # Use fixed height
        spacing=spacing,
        rotations=rotations,
        iterations=optimization_iterations
    )
    
    # Run optimization
    pattern_pieces = optimizer.optimize(pattern_pieces)
    
    # Calculate the actual used area (smallest enclosing rectangle)
    actual_used_width, actual_used_height = calculate_smallest_enclosing_rectangle(pattern_pieces)
    actual_used_area = actual_used_width * actual_used_height
    
    # Step 5: Analyze scraps in OPTIMIZED layout
    logger.info("Analyzing optimized layout...")
    optimized_analyzer = ScrapAnalyzer(fabric_width, fabric_height)  # Use fixed dimensions
    optimized_analysis = optimized_analyzer.analyze(pattern_pieces)
    
    # Step 6: Calculate statistics and improvements
    logger.info("Calculating statistics and improvements")
    
    # Calculate fabric utilization
    original_utilization = (total_pattern_area / original_fabric_area) * 100 if original_fabric_area > 0 else 0
    optimized_fabric_area = fabric_width * fabric_height
    optimized_utilization = (total_pattern_area / optimized_fabric_area) * 100 if optimized_fabric_area > 0 else 0
    
    # Calculate fabric savings
    fabric_saved = original_fabric_area - optimized_fabric_area
    fabric_saved_percent = (fabric_saved / original_fabric_area) * 100 if original_fabric_area > 0 else 0
    
    # Count rotated pieces
    rotated_pieces = sum(1 for piece in pattern_pieces 
                      if 'rotation_angle' in piece and piece['rotation_angle'] != 0)
    
    # Calculate usable fabric metrics
    total_usable_fabric_gained = float(optimized_analysis['total_usable_area']) - float(original_analysis['total_usable_area'])
    usable_fabric_gain_percent = (total_usable_fabric_gained / original_analysis['total_usable_fabric']) * 100 if original_analysis['total_usable_fabric'] > 0 else 0
    
    # Calculate improvement metrics
    scrap_reduction = float(original_analysis['total_scrap_area']) - float(optimized_analysis['total_scrap_area'])
    scrap_reduction_percent = (scrap_reduction / original_analysis['total_scrap_area']) * 100 if original_analysis['total_scrap_area'] > 0 else 0
    
    # Rectangle improvement variables
    rectangle_improvement = optimized_analysis['largest_rectangle_area'] - original_analysis['largest_rectangle_area']
    rectangle_improvement_percent = (rectangle_improvement / original_analysis['largest_rectangle_area']) * 100 if original_analysis['largest_rectangle_area'] > 0 else 0
    
    # Quality improvement variable
    quality_improvement = optimized_analysis['scrap_quality_index'] - original_analysis['scrap_quality_index']

    # Calculate net fabric consumption
    original_net_fabric_used = original_fabric_area - original_analysis['total_usable_area']
    optimized_net_fabric_used = actual_used_area - optimized_analysis['total_usable_area']
    net_fabric_reduction = original_net_fabric_used - optimized_net_fabric_used
    net_fabric_reduction_percent = (net_fabric_reduction / original_net_fabric_used) * 100 if original_net_fabric_used > 0 else 0

    # Update statistics list with new metrics
    stats = [
        "== Overall Fabric Required ==",
        f"Fixed fabric dimensions: {fabric_width} × {fabric_height} pixels = {fabric_width * fabric_height:,} sq. pixels",
        f"Original layout used area: {orig_width} × {orig_height} pixels = {original_fabric_area:,} sq. pixels",
        f"Optimized layout used area: {actual_used_width:.0f} × {actual_used_height:.0f} pixels = {actual_used_area:,} sq. pixels",
        f"Actual area reduction: {original_fabric_area - actual_used_area:,} sq. pixels ({(original_fabric_area - actual_used_area) / original_fabric_area * 100:.2f}%)",
    ]
    
    # Add seam allowance info if applied
    if seam_allowance > 0:
        stats.append(f"Seam allowance applied: {seam_allowance} pixels")
    
    stats.extend([
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
        f"Total reusable fabric gained: {float(optimized_analysis['total_usable_area']) - float(original_analysis['total_usable_area']):,.2f} sq. pixels",
        f"Reusable fabric improvement: {quality_improvement:.2f}%",
        "",
        "== Other Stats ==",
        f"Pieces placed: {placed_count}/{len(pattern_pieces)}",
        f"Pieces rotated: {rotated_pieces}/{len(pattern_pieces)}",
        f"Processing time: {time.time() - start_time:.2f} seconds"
    ])

    area_reduction = original_fabric_area - actual_used_area
    area_reduction_percent = (area_reduction / original_fabric_area) * 100 if original_fabric_area > 0 else 0

    # Update the optimization summary
    if area_reduction_percent > 0:
        trade_off = f"This optimization REDUCES total fabric needed by {net_fabric_reduction_percent:.2f}%, which saves material costs."
    else:
        trade_off = f"This optimization INCREASES total fabric area by {abs(net_fabric_reduction_percent):.2f}%, but provides more reusable leftover fabric."

    if quality_improvement > 0:
        quality = f"The reusable fabric quality improved by {quality_improvement:.2f}%, meaning less small scrap waste."
    else:
        quality = f"The reusable fabric quality decreased slightly, but is compensated by using less total fabric."

    stats.extend([
        "",
        "== Optimization Summary ==",
        trade_off,
        quality,
        f"Overall optimization effectiveness: {'Good' if fabric_saved_percent > 0 or quality_improvement > 2 else 'Moderate'}"
    ])
    
    # Print statistics
    for stat in stats:
        logger.info(stat)
    
    # Step 7: Generate output files
    svg_path = f"{output_prefix}.svg"
    pdf_path = f"{output_prefix}.pdf"
    
    # Create SVG with scrap analysis
    create_svg(pattern_pieces, fabric_width, fabric_height, optimized_analysis, svg_path)
    
    # Create PDF with comparison - use original_pieces without seam allowance for comparison
    create_pdf(pattern_pieces, original_pieces, original_width, original_height, 
               fabric_width, fabric_height, stats, 
               original_analysis, optimized_analysis, pdf_path)
    
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
        'optimized_analysis': optimized_analysis,
        'improvements': {
            'scrap_reduction_percent': scrap_reduction_percent,
            'rectangle_improvement_percent': rectangle_improvement_percent,
            'quality_improvement': quality_improvement
        }
    }


def calculate_smallest_enclosing_rectangle(pattern_pieces):
    """Calculate the smallest rectangle that contains all pattern pieces with safety margin."""
    if not pattern_pieces:
        return 0, 0
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = 0
    max_y = 0
    
    # Track if we have valid coordinates
    has_valid_coords = False
    
    for piece in pattern_pieces:
        if 'new_x' not in piece or 'new_y' not in piece:
            continue
            
        # Get positioned polygon
        try:
            # Use points with position offset applied
            points = piece['points']
            dx = piece['new_x'] - piece['x']
            dy = piece['new_y'] - piece['y']
            
            # Check if rotation is needed
            if 'rotation_angle' in piece and piece['rotation_angle'] != 0:
                # Calculate centroid
                centroid_x = sum(p[0] for p in points) / len(points)
                centroid_y = sum(p[1] for p in points) / len(points)
                angle_rad = math.radians(piece['rotation_angle'])
                
                # Rotate points around centroid
                adjusted_points = []
                for px, py in points:
                    # Translate to origin
                    px_centered = px - centroid_x
                    py_centered = py - centroid_y
                    
                    # Rotate
                    px_rotated = (px_centered * math.cos(angle_rad) - 
                                 py_centered * math.sin(angle_rad))
                    py_rotated = (px_centered * math.sin(angle_rad) + 
                                 py_centered * math.cos(angle_rad))
                    
                    # Translate back
                    adjusted_points.append((
                        px_rotated + centroid_x + dx,
                        py_rotated + centroid_y + dy
                    ))
            else:
                # Just translate
                adjusted_points = [(p[0] + dx, p[1] + dy) for p in points]
            
            # Update min and max coordinates
            for px, py in adjusted_points:
                if isinstance(px, (int, float)) and isinstance(py, (int, float)):
                    min_x = min(min_x, px)
                    min_y = min(min_y, py)
                    max_x = max(max_x, px)
                    max_y = max(max_y, py)
                    has_valid_coords = True
                
        except Exception as e:
            logger.warning(f"Error calculating bounds: {e}")
            # Fallback to bounding box if points manipulation fails
            if 'width' in piece and 'height' in piece:
                piece_min_x = float(piece['new_x'])
                piece_min_y = float(piece['new_y'])
                piece_max_x = piece_min_x + float(piece['width'])
                piece_max_y = piece_min_y + float(piece['height'])
                
                min_x = min(min_x, piece_min_x)
                min_y = min(min_y, piece_min_y)
                max_x = max(max_x, piece_max_x)
                max_y = max(max_y, piece_max_y)
                has_valid_coords = True
    
    # Handle edge case
    if not has_valid_coords:
        return 0, 0
        
    # Add margin for clarity
    margin = 10  # Larger margin for safety
    width = max(0, max_x - min_x + margin*2)
    height = max(0, max_y - min_y + margin*2)
    
    return width, height


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Zero Waste Pattern Layout Optimizer - Hybrid Approach with Scrap Analysis')
    
    parser.add_argument('input_pdf', help='Path to input PDF with pattern pieces')
    parser.add_argument('--output', '-o', default='optimized_layout', help='Output file prefix')
    parser.add_argument('--width', '-w', type=int, help='Maximum fabric width in pixels')
    parser.add_argument('--spacing', '-s', type=float, default=DEFAULT_SPACING,
                        help=f'Spacing between parts in pixels (default: {DEFAULT_SPACING})')
    parser.add_argument('--iterations', '-i', type=int, default=DEFAULT_OPTIMIZATION_ITERATIONS,
                        help=f'Number of optimization iterations (default: {DEFAULT_OPTIMIZATION_ITERATIONS})')
    parser.add_argument('--seam-allowance', '-a', type=int, default=10,
                        help=f'Seam allowance in pixels (default: 10, 0 to disable)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run optimization with scrap analysis
    optimize_pattern_layout(
        args.input_pdf, args.output,
        fabric_width=args.width,
        spacing=args.spacing,
        optimization_iterations=args.iterations,
        seam_allowance=args.seam_allowance
    )


if __name__ == "__main__":
    main()