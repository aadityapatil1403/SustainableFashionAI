import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import json
from pattern_vae_torch import VAE, train_vae, test_vae, visualize_reconstructions, generate_samples, convert_image_to_polygon

def main():
    parser = argparse.ArgumentParser(description='Train VAE for pattern generation')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Dimension of latent space')
    parser.add_argument('--input-dim', type=int, default=128,
                        help='Dimension of input images')
    parser.add_argument('--input-dir', type=str, default='hoodie_processed_patterns',
                        help='Directory containing processed pattern data')
    parser.add_argument('--output-dir', type=str, default='hoodie_vae_output_improved',
                        help='Directory to save model and generated patterns')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                        help='Learning rate for optimizer')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data_file = os.path.join(args.input_dir, "pattern_images.npy")
    print(f"Loading data from {data_file}")
    pattern_images = np.load(data_file)
    
    # Resize images to higher resolution if needed
    if pattern_images.shape[1] != args.input_dim:
        from skimage.transform import resize
        print(f"Resizing images from {pattern_images.shape[1:3]} to {(args.input_dim, args.input_dim)}")
        resized_images = []
        for img in pattern_images:
            resized = resize(img, (args.input_dim, args.input_dim), 
                            order=1, preserve_range=True, anti_aliasing=True)
            resized_images.append(resized)
        pattern_images = np.array(resized_images)
    
    # Convert to PyTorch tensors
    pattern_images = torch.FloatTensor(pattern_images)
    pattern_images = pattern_images.permute(0, 3, 1, 2)  # Change from [N, H, W, C] to [N, C, H, W]
    
    # Split data
    train_data, test_data = train_test_split(pattern_images, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Create and train VAE
    model = VAE(input_dim=args.input_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_vae(model, train_loader, optimizer, device, epoch)
        test_loss = test_vae(model, test_loader, device)
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model with test loss: {best_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with test loss: {best_loss:.4f}")
    
    # Save model
    model_path = os.path.join(args.output_dir, 'pattern_vae_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # Visualize reconstructions
    recon_path = os.path.join(args.output_dir, 'reconstructions.png')
    visualize_reconstructions(
        model, 
        test_loader, 
        device, 
        n=10, 
        output_file=recon_path
    )
    
    # Generate new patterns with different temperatures
    n_samples = 10
    temperatures = [0.7, 0.8, 0.9, 1.0]
    
    for temp in temperatures:
        print(f"Generating {n_samples} new patterns with temperature {temp}...")
        
        # Generate samples with temperature
        generated_samples = []
        model.eval()
        with torch.no_grad():
            # Sample from latent space with temperature
            z = torch.randn(n_samples, model.latent_dim).to(device) * temp
            samples = model.decode(z)
            generated_samples = samples.cpu().numpy()
        
        # Visualize generated samples
        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
        for i in range(n_samples):
            axes[i].imshow(generated_samples[i].squeeze(), cmap='gray')
            axes[i].axis('off')
        
        plt.tight_layout()
        gen_path = os.path.join(args.output_dir, f'generated_patterns_temp{temp:.1f}.png')
        plt.savefig(gen_path)
        plt.close()
        print(f"Saved generated patterns visualization to {gen_path}")
        
        # Convert to polygons
        generated_polygons = []
        for i, sample in enumerate(generated_samples):
            # Use a higher threshold for clearer boundaries
            polygon = convert_image_to_polygon(sample, threshold=0.5)
            if polygon and len(polygon) >= 4:  # Only add valid polygons with at least 4 points
                generated_polygons.append({
                    'id': f'generated_{i}_temp{temp:.1f}',
                    'points': polygon,
                    'is_synthetic': True
                })
        
        # Save to JSON
        json_path = os.path.join(args.output_dir, f'generated_patterns_temp{temp:.1f}.json')
        with open(json_path, 'w') as f:
            json.dump({'pattern_pieces': generated_polygons}, f, indent=2)
        
        print(f"Saved {len(generated_polygons)} generated patterns to {json_path}")

if __name__ == "__main__":
    main()
