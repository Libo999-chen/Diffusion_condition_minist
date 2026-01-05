import torch
import random

import torch
import random

def generate_random_mask(x):
    """
    Generate a random mask for the given image tensor `x`, where the mask is defined 
    by randomly chosen `x1` and `y1` coordinates.

    Parameters:
    - x: Tensor of shape (batch_size, channels, height, width)

    Returns:
    - mask: Tensor of the same shape as `x` with the mask applied
    """

    # Get the dimensions of the image
    _, _, height, width = x.shape
    
    # Randomly select x1 and y1 within the image dimensions
    x1 = random.randint(0, width-1)
    print(x1)
    y1 = random.randint(0, height-1)
    print(y1)

    # Generate mask
    mask = torch.ones_like(x)
    
    # Define the region within the mask using (0, 0), (x1, y1)
    mask[:, :, :y1, :x1] = 0

    return x * mask  # Apply mask to the image



# Create a random tensor with shape (batch_size=1, channels=3, height=32, width=32)
x = torch.randn(1, 3, 5, 5)
mask = torch.ones_like(x)
mask[:, :, :, 3:] = 0. 
y1= x * mask  

# Call the function with the input tensor
masked_x = generate_random_mask(x)


# Optionally, print the masked image for visual inspection (only for small images)
print("Original Image (Sample):", x[0, 0, :5, :5])  # Print a small section of the original image
print("Masked Image (Sample):", masked_x[0, 0, :5, :5])  # Print a small section of the masked image
print("Masked Image (Sample):", y1[0, 0, :5, :5]) 