# SustainableFashionAI

This repository contains code for an AI-powered system that reduces fabric waste in fashion production through intelligent pattern recognition, layout optimization, and automated technical documentation generation.

## Repository Structure

### SketchToVector
Tools for converting fashion design sketches to vector formats:
- `sketchtovector.py` - Converts sketches to SVG vector format
- `sketch_to_techpack.py` - Enhanced version with tech pack preparation
- `digitalSketch_to_vector.py` - Optimized for digital sketch inputs

### PatternOptimizer
Pattern layout optimization engines to minimize fabric waste:
- `hybrid_pattern_optimizer.py` - Advanced algorithm with polygon refinement
- `bin_packing_optimizer.py` - Simpler bin packing implementation
- `nfp_optimizer.py` - No-Fit Polygon based layout optimization

### TechPackGeneration
Automated technical documentation generation:
- `techpacksummary.py` - Generates tech pack documentation using LLM models
- `garment_detection.py` - Computer vision for garment component detection

### VAE
Generative models for creating and manipulating pattern designs:
- `pattern_vae_torch.py` - Core VAE implementation
- `train_hoodie_vae_improved.py` - Training script for hoodie patterns
- `prepare_pattern_data.py` - Processes pattern data for training
- `integrate_synthetic_patterns.py` - Connects generated patterns to optimizer

### Frontend
UI components and visualization tools (directory structure)

### Sketches
Example design sketches and artwork (directory structure)

### Utils
Utility scripts for pattern processing:
- `extract_hoodie_patterns.py` - Extracts pattern pieces from PDF files
- `convert_json_to_pdf.py` - Converts pattern data to PDF format
