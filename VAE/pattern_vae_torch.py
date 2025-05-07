import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from shapely.geometry import Polygon
import json
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class VAE(nn.Module):
    def __init__(self, input_dim=64, latent_dim=32):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Calculate sizes for the network
        # For 128x128 input: 64->32->16->8
        self.conv1_out = input_dim // 2
        self.conv2_out = self.conv1_out // 2
        self.conv3_out = self.conv2_out // 2
        self.conv4_out = self.conv3_out // 2
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            # Layer 1: input_dim x input_dim x 1 -> conv1_out x conv1_out x 32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 2: conv1_out x conv1_out x 32 -> conv2_out x conv2_out x 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 3: conv2_out x conv2_out x 64 -> conv3_out x conv3_out x 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Layer 4: conv3_out x conv3_out x 128 -> conv4_out x conv4_out x 256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Calculate flattened size
        self.flat_size = 256 * self.conv4_out * self.conv4_out
        
        # Latent space mapping
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # Decoder initial layer
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)
        
        # Decoder
        self.decoder_conv = nn.Sequential(
            # Layer 1: flat_size -> conv4_out x conv4_out x 256
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 2: conv4_out*2 x conv4_out*2 x 128 -> conv3_out*2 x conv3_out*2 x 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3: conv3_out*2 x conv3_out*2 x 64 -> conv2_out*2 x conv2_out*2 x 32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 4: conv2_out*2 x conv2_out*2 x 32 -> conv1_out*2 x conv1_out*2 x 1
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1, self.flat_size)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, self.conv4_out, self.conv4_out)
        x = self.decoder_conv(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    VAE loss function: reconstruction loss + KL divergence
    """
    # Reconstruction loss (binary cross entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def train_vae(model, train_loader, optimizer, device, epoch, log_interval=10):
    """
    Train the VAE for one epoch
    """
    model.train()
    train_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    return train_loss / len(train_loader.dataset)

def test_vae(model, test_loader, device):
    """
    Test the VAE
    """
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss

def generate_samples(model, n_samples, device):
    """
    Generate new samples from the VAE
    """
    model.eval()
    with torch.no_grad():
        # Sample from latent space
        z = torch.randn(n_samples, model.latent_dim).to(device)
        samples = model.decode(z)
    
    return samples.cpu().numpy()

def preprocess_pattern_data(pattern_pieces, target_size=(64, 64)):
    """
    Convert pattern pieces to binary images for VAE training
    
    Args:
        pattern_pieces: List of pattern pieces, each with 'points' key
        target_size: Size of the output images
    
    Returns:
        Numpy array of binary images
    """
    images = []
    
    for piece in pattern_pieces:
        # Create a blank image
        img = np.zeros(target_size, dtype=np.uint8)
        
        # Get points and convert to numpy array of integers
        points = np.array(piece['points'], dtype=np.int32)
        
        # Draw the polygon
        cv2.fillPoly(img, [points], 255)
        
        # Normalize
        img = img / 255.0
        
        images.append(img)
    
    # Convert to numpy array with channel dimension
    return np.array(images)[..., np.newaxis]

def load_pattern_data(pattern_files):
    """
    Load pattern data from JSON files
    
    Args:
        pattern_files: List of pattern JSON files
    
    Returns:
        List of pattern pieces
    """
    all_pieces = []
    
    for file in pattern_files:
        with open(file, 'r') as f:
            pattern_data = json.load(f)
            
            # Extract pattern pieces
            pieces = pattern_data.get('pattern_pieces', [])
            all_pieces.extend(pieces)
    
    return all_pieces

def convert_image_to_polygon(image, threshold=0.5, min_area=10, epsilon_factor=0.005):
    """
    Convert a generated image back to polygon points
    
    Args:
        image: Generated binary image
        threshold: Threshold for binarization
        min_area: Minimum contour area to consider
        epsilon_factor: Factor for polygon approximation (Douglas-Peucker algorithm)
    
    Returns:
        List of polygon points
    """
    # Convert to binary image
    binary = (image > threshold).astype(np.uint8) * 255
    binary = binary.squeeze()  # Remove channel dimension
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Filter contours by area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours:
        return []
    
    # Get the largest contour
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Simplify the contour using Douglas-Peucker algorithm
    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to polygon points
    polygon_points = approx.reshape(-1, 2).tolist()
    
    return polygon_points

def visualize_reconstructions(model, test_loader, device, n=10, output_file=None):
    """
    Visualize original and reconstructed patterns
    """
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, _, _ = model(data)
            
            # Only use the first batch
            if i == 0:
                n = min(n, data.size(0))
                comparison = torch.cat([data[:n], recon_batch[:n]])
                
                # Convert to numpy and reshape
                comparison = comparison.cpu().numpy()
                
                # Create figure
                fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
                
                for j in range(n):
                    # Original
                    axes[0, j].imshow(comparison[j].squeeze(), cmap='gray')
                    axes[0, j].axis('off')
                    
                    # Reconstructed
                    axes[1, j].imshow(comparison[n + j].squeeze(), cmap='gray')
                    axes[1, j].axis('off')
                
                plt.tight_layout()
                
                if output_file:
                    plt.savefig(output_file)
                    print(f"Saved reconstructions to {output_file}")
                
                plt.close()
                break

def main():
    # Parameters
    input_dim = 64
    latent_dim = 32
    batch_size = 16
    epochs = 20
    output_dir = "vae_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_file = "processed_patterns/pattern_images.npy"
    print(f"Loading data from {data_file}")
    pattern_images = np.load(data_file)
    
    # Convert to PyTorch tensors
    pattern_images = torch.FloatTensor(pattern_images)
    pattern_images = pattern_images.permute(0, 3, 1, 2)  # Change from [N, H, W, C] to [N, C, H, W]
    
    # Split data
    train_data, test_data = train_test_split(pattern_images, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Create and train VAE
    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_vae(model, train_loader, optimizer, device, epoch)
        test_loss = test_vae(model, test_loader, device)
    
    # Save model
    model_path = os.path.join(output_dir, 'pattern_vae_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # Visualize reconstructions
    visualize_reconstructions(
        model, 
        test_loader, 
        device, 
        n=10, 
        output_file=os.path.join(output_dir, 'reconstructions.png')
    )
    
    # Generate new patterns
    n_samples = 10
    print(f"Generating {n_samples} new patterns...")
    generated_samples = generate_samples(model, n_samples, device)
    
    # Visualize generated samples
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
    for i in range(n_samples):
        axes[i].imshow(generated_samples[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generated_patterns.png'))
    plt.close()
    
    # Convert to polygons
    generated_polygons = []
    for i, sample in enumerate(generated_samples):
        polygon = convert_image_to_polygon(sample)
        if polygon:
            generated_polygons.append({
                'id': f'generated_{i}',
                'points': polygon,
                'is_synthetic': True
            })
    
    # Save to JSON
    with open(os.path.join(output_dir, 'generated_patterns.json'), 'w') as f:
        json.dump({'pattern_pieces': generated_polygons}, f, indent=2)
    
    print(f"Saved {len(generated_polygons)} generated patterns to {os.path.join(output_dir, 'generated_patterns.json')}")

if __name__ == "__main__":
    main()
