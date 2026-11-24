"""
Train Mnemosyne VAE on ZeroTraceGPT Images
DeathAIAUS Business System
"""

import sys
import os
import numpy as np
import asyncio
import logging
from datetime import datetime

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Mnemosyne_VAE'))

# Dynamic import of Mnemosyne
import importlib.util
mnemosyne_path = os.path.join(os.path.dirname(__file__), 'Mnemosyne_VAE', '37_MNEMOSYNE_DeathAIAUS.py')
spec = importlib.util.spec_from_file_location("mnemosyne_module", mnemosyne_path)
mnemosyne_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mnemosyne_module)

MnemosyneDeathAIAUS = mnemosyne_module.MnemosyneDeathAIAUS
MnemosyneConfig = mnemosyne_module.MnemosyneConfig
MnemosyneInput = mnemosyne_module.MnemosyneInput
MnemosyneDataType = mnemosyne_module.MnemosyneDataType
MnemosyneArchitectureType = mnemosyne_module.MnemosyneArchitectureType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def train_mnemosyne():
    """Train Mnemosyne VAE on real ZeroTraceGPT images"""
    
    print("="*70)
    print("DeathAIAUS Mnemosyne VAE Training")
    print("Real Image Data from ZeroTraceGPT")
    print("="*70)
    
    # Load training data
    print("\nLoading training data...")
    data_path = os.path.join("Training_Data", "zerotrace_28x28.npz")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    data = np.load(data_path)
    images = data['images']
    
    logger.info(f"Loaded {len(images)} images from {data_path}")
    logger.info(f"Image shape: {images.shape}")
    logger.info(f"Data range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Initialize Mnemosyne with RGB image configuration
    print("\nInitializing Mnemosyne VAE...")
    config = MnemosyneConfig(
        data_type=MnemosyneDataType.IMAGE,
        architecture_type=MnemosyneArchitectureType.CONVOLUTIONAL,
        input_shape=(28, 28, 3),  # RGB images
        latent_dim=32,  # Larger latent space for real images
        hidden_dims=[512, 256, 128],  # Deeper network
        learning_rate=0.0001,  # Lower learning rate for stability
        beta=1.0
    )
    
    mnemosyne = MnemosyneDeathAIAUS(config)
    
    logger.info("Mnemosyne initialized with configuration:")
    logger.info(f"  Input shape: {config.input_shape}")
    logger.info(f"  Latent dimension: {config.latent_dim}")
    logger.info(f"  Hidden dimensions: {config.hidden_dims}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    
    # Training configuration
    num_epochs = 50  # More epochs for real data
    batch_size = 32  # Good batch size for 10,707 images
    validation_split = 0.2
    
    logger.info(f"\nTraining configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Validation split: {validation_split}")
    
    # Train the model
    print("\nStarting training...")
    print("-"*70)
    
    start_time = datetime.now()
    result = await mnemosyne.mnemosyne.train(
        train_data=images,
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )
    end_time = datetime.now()
    
    training_duration = (end_time - start_time).total_seconds()
    
    print("-"*70)
    print("\nTraining Results:")
    print(f"  Status: {'SUCCESS' if result['training_completed'] else 'FAILED'}")
    print(f"  Epochs completed: {result['epochs_completed']}")
    print(f"  Final loss: {result.get('final_loss', 'N/A'):.4f}")
    print(f"  Final reconstruction loss: {result.get('final_recon_loss', 'N/A'):.4f}")
    print(f"  Final KL loss: {result.get('final_kl_loss', 'N/A'):.4f}")
    print(f"  Training duration: {training_duration:.2f} seconds")
    
    # Test generation
    print("\nTesting trained model...")
    print("Generating new samples...")
    
    gen_result = await mnemosyne.generate(num_samples=4, temperature=1.0)
    
    if gen_result.generated_data is not None:
        print(f"  Generated shape: {gen_result.generated_data.shape}")
        print(f"  Generated range: [{gen_result.generated_data.min():.3f}, {gen_result.generated_data.max():.3f}]")
    
    # Test reconstruction
    print("\nTesting reconstruction...")
    test_image = images[0]
    
    encode_result = await mnemosyne.encode(test_image)
    if encode_result.latent_mean is not None:
        print(f"  Encoded latent shape: {encode_result.latent_mean.shape}")
        
        decode_result = await mnemosyne.decode(encode_result.latent_mean)
        if decode_result.reconstructed_data is not None:
            print(f"  Decoded shape: {decode_result.reconstructed_data.shape}")
            
            # Calculate reconstruction error
            recon_error = np.mean((test_image - decode_result.reconstructed_data) ** 2)
            print(f"  Reconstruction MSE: {recon_error:.4f}")
    
    # Save the trained model
    save_path = os.path.join("Mnemosyne_VAE", "mnemosyne_trained_zerotrace.pth")
    print(f"\nSaving trained model to {save_path}...")
    
    mnemosyne.mnemosyne.save_model(save_path)
    
    print(f"Model saved successfully!")
    
    # Save training summary
    summary_path = os.path.join("Training_Logs", "mnemosyne_training_summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DeathAIAUS Mnemosyne VAE Training Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Training Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Duration: {training_duration:.2f} seconds\n\n")
        f.write("Configuration:\n")
        f.write(f"  Data Type: IMAGE (RGB)\n")
        f.write(f"  Input Shape: {config.input_shape}\n")
        f.write(f"  Latent Dimension: {config.latent_dim}\n")
        f.write(f"  Hidden Dimensions: {config.hidden_dims}\n")
        f.write(f"  Learning Rate: {config.learning_rate}\n")
        f.write(f"  Beta: {config.beta}\n\n")
        f.write("Training Data:\n")
        f.write(f"  Source: ZeroTraceGPT Images\n")
        f.write(f"  Total Images: {len(images)}\n")
        f.write(f"  Image Shape: {images.shape[1:]}\n\n")
        f.write("Training Parameters:\n")
        f.write(f"  Epochs: {num_epochs}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Validation Split: {validation_split}\n\n")
        f.write("Results:\n")
        f.write(f"  Training Completed: {result['training_completed']}\n")
        f.write(f"  Epochs Completed: {result['epochs_completed']}\n")
        f.write(f"  Final Loss: {result.get('final_loss', 'N/A')}\n")
        f.write(f"  Final Reconstruction Loss: {result.get('final_recon_loss', 'N/A')}\n")
        f.write(f"  Final KL Loss: {result.get('final_kl_loss', 'N/A')}\n\n")
        f.write("Model Path:\n")
        f.write(f"  {save_path}\n\n")
        f.write("="*70 + "\n")
        f.write("DeathAIAUS - Premium Bot Solutions\n")
        f.write("https://deathaiaustralia.com.au\n")
        f.write("="*70 + "\n")
    
    print(f"Training summary saved to {summary_path}")
    
    print("\n" + "="*70)
    print("Mnemosyne VAE Training Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(train_mnemosyne())

