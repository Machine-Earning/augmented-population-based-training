#!/usr/bin/env python3
"""
Download real MNIST and Fashion MNIST datasets using torchvision
"""

import numpy as np
import os
from pathlib import Path

def create_apbt_format(images, labels, output_path, is_attr=False):
    """Convert to APBT format"""
    
    if is_attr:
        # Create attribute file
        with open(output_path, 'w') as f:
            # Input attributes (784 pixels, each 0-1)
            for i in range(784):
                f.write(f'pixel{i} 0 1\n')
            f.write('\n')
            # Output: 1 categorical attribute with 10 classes (0-9)
            # This gets one-hot encoded to 10 binary output units in the neural network
            f.write('class 0 1 2 3 4 5 6 7 8 9\n')
        return
    
    # Normalize and flatten images
    images_flat = images.reshape(images.shape[0], -1) / 255.0
    
    # Write data file
    with open(output_path, 'w') as f:
        for img, label in zip(images_flat, labels):
            # Write flattened pixel values
            pixel_str = ' '.join([f'{p:.6f}' for p in img])
            # Write label
            f.write(f'{pixel_str} {label}\n')

def main():
    print("="*70)
    print("Downloading Real MNIST and Fashion MNIST Datasets")
    print("="*70)
    
    try:
        # Import torchvision
        print("\nImporting torchvision...")
        import torchvision
        from torchvision import datasets, transforms
        print("✓ Successfully imported")
    except ImportError:
        print("\n✗ torchvision not found. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'torchvision'])
        import torchvision
        from torchvision import datasets, transforms
    
    # Create directories
    Path('data/mnist').mkdir(parents=True, exist_ok=True)
    Path('data/fashion_mnist').mkdir(parents=True, exist_ok=True)
    Path('data/samples').mkdir(parents=True, exist_ok=True)
    Path('data/temp').mkdir(parents=True, exist_ok=True)
    
    # Download MNIST
    print("\n1. Downloading MNIST...")
    try:
        train_dataset = datasets.MNIST(root='data/temp', train=True, download=True)
        test_dataset = datasets.MNIST(root='data/temp', train=False, download=True)
        
        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()
        
        print(f"  ✓ Downloaded {len(train_images)} training and {len(test_images)} test samples")
        
        # Create APBT format files
        print("\n  Converting to APBT format...")
        create_apbt_format(None, None, 'data/mnist/mnist-attr.txt', is_attr=True)
        print("  ✓ Created attribute file")
        
        # Use subset for faster training (adjust for speed vs accuracy tradeoff)
        subset_size_train = min(3000, len(train_images))  # Reduced for faster training
        subset_size_test = min(600, len(test_images))     # Reduced for faster evaluation
        
        create_apbt_format(train_images[:subset_size_train], train_labels[:subset_size_train], 
                          'data/mnist/mnist-train.txt')
        print(f"  ✓ Created training file ({subset_size_train} samples)")
        
        create_apbt_format(test_images[:subset_size_test], test_labels[:subset_size_test], 
                          'data/mnist/mnist-test.txt')
        print(f"  ✓ Created test file ({subset_size_test} samples)")
        
        # Save sample images (5 random samples from different classes)
        sample_indices = []
        for digit in range(5):
            # Get all indices for this digit and randomly select one
            digit_indices = np.where(test_labels == digit)[0]
            idx = np.random.choice(digit_indices)
            sample_indices.append(idx)
        
        sample_images = test_images[sample_indices]
        sample_labels = test_labels[sample_indices]
        
        np.save('data/samples/mnist_samples.npy', {
            'images': sample_images,
            'labels': sample_labels
        }, allow_pickle=True)
        print("  ✓ Saved 5 sample images")
        
    except Exception as e:
        print(f"  ✗ Error downloading MNIST: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Download Fashion MNIST
    print("\n2. Downloading Fashion MNIST...")
    try:
        train_dataset = datasets.FashionMNIST(root='data/temp', train=True, download=True)
        test_dataset = datasets.FashionMNIST(root='data/temp', train=False, download=True)
        
        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()
        
        print(f"  ✓ Downloaded {len(train_images)} training and {len(test_images)} test samples")
        
        # Create APBT format files
        print("\n  Converting to APBT format...")
        create_apbt_format(None, None, 'data/fashion_mnist/fashion-mnist-attr.txt', is_attr=True)
        print("  ✓ Created attribute file")
        
        # Use subset for faster training (adjust for speed vs accuracy tradeoff)
        subset_size_train = min(3000, len(train_images))  # Reduced for faster training
        subset_size_test = min(600, len(test_images))     # Reduced for faster evaluation
        
        create_apbt_format(train_images[:subset_size_train], train_labels[:subset_size_train], 
                          'data/fashion_mnist/fashion-mnist-train.txt')
        print(f"  ✓ Created training file ({subset_size_train} samples)")
        
        create_apbt_format(test_images[:subset_size_test], test_labels[:subset_size_test], 
                          'data/fashion_mnist/fashion-mnist-test.txt')
        print(f"  ✓ Created test file ({subset_size_test} samples)")
        
        # Save sample images (5 random samples from different classes)
        sample_indices = []
        for label in range(5):
            # Get all indices for this label and randomly select one
            label_indices = np.where(test_labels == label)[0]
            idx = np.random.choice(label_indices)
            sample_indices.append(idx)
        
        sample_images = test_images[sample_indices]
        sample_labels = test_labels[sample_indices]
        
        np.save('data/samples/fashion_mnist_samples.npy', {
            'images': sample_images,
            'labels': sample_labels
        }, allow_pickle=True)
        print("  ✓ Saved 5 sample images")
        
    except Exception as e:
        print(f"  ✗ Error downloading Fashion MNIST: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("Dataset download complete!")
    print("="*70)
    print("\nFiles created:")
    print("  ✓ data/mnist/mnist-attr.txt")
    print("  ✓ data/mnist/mnist-train.txt (3,000 samples)")
    print("  ✓ data/mnist/mnist-test.txt (600 samples)")
    print("  ✓ data/fashion_mnist/fashion-mnist-attr.txt")
    print("  ✓ data/fashion_mnist/fashion-mnist-train.txt (3,000 samples)")
    print("  ✓ data/fashion_mnist/fashion-mnist-test.txt (600 samples)")
    print("  ✓ data/samples/mnist_samples.npy (5 sample images)")
    print("  ✓ data/samples/fashion_mnist_samples.npy (5 sample images)")
    print("\nNote: Using subsets for faster training (3K train, 600 test).")
    print("Original datasets: 60,000 training + 10,000 test samples each")
    print("To use more data, edit subset_size_train and subset_size_test in download_mnist.py")
    print("\nClass labels for Fashion MNIST:")
    print("  0: T-shirt/top")
    print("  1: Trouser")
    print("  2: Pullover")
    print("  3: Dress")
    print("  4: Coat")
    print("  5: Sandal")
    print("  6: Shirt")
    print("  7: Sneaker")
    print("  8: Bag")
    print("  9: Ankle boot")
    print("="*70)

if __name__ == '__main__':
    main()
