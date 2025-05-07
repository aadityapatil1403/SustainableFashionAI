import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon as MplPolygon
import random

def load_pattern_set(json_file):
    """
    Load pattern set from JSON file
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data

def convert_to_pdf(pattern_data, output_pdf):
    """
    Convert pattern data to PDF format
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    # Set up the plot
    ax.set_aspect('equal')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.axis('off')
    
    # Colors for different pattern pieces
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot each pattern piece
    pattern_pieces = pattern_data.get('pattern_pieces', [])
    for i, piece in enumerate(pattern_pieces):
        points = np.array(piece['points'])
        
        # Create polygon
        color = colors[i % len(colors)]
        polygon = MplPolygon(points, closed=True, fill=True, 
                             facecolor=color, alpha=0.7, 
                             edgecolor='black', linewidth=1)
        ax.add_patch(polygon)
        
        # Add piece ID as text
        centroid_x = np.mean(points[:, 0])
        centroid_y = np.mean(points[:, 1])
        ax.text(centroid_x, centroid_y, piece.get('id', f'Piece {i+1}'), 
                ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Save to PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    
    plt.close(fig)
    print(f"Saved pattern set to {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description='Convert JSON pattern set to PDF')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the pattern set JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output PDF file')
    
    args = parser.parse_args()
    
    # Load pattern set
    pattern_data = load_pattern_set(args.input)
    
    # Convert to PDF
    convert_to_pdf(pattern_data, args.output)

if __name__ == "__main__":
    main()
