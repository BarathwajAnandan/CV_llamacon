import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_matrix(file_path='./load_matrix.npy'):
    """
    Load a NumPy array from a file.
    
    Args:
        file_path (str): Path to the NumPy file
        
    Returns:
        np.ndarray: The loaded NumPy array
    """
    try:
        return np.load(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None

def plot_expert_load_heatmap(load_matrix, num_experts=16, num_layers=48, figsize=(14, 10), 
                            cmap="YlGnBu", title="Expert Token Load Heatmap Across MoE Layers"):
    """
    Create a heatmap visualization of expert token loads across MoE layers.
    
    Args:
        load_matrix (list or np.ndarray): Matrix of expert loads, shape (num_layers, num_experts)
        num_experts (int): Number of experts
        num_layers (int): Number of MoE layers
        figsize (tuple): Figure size (width, height)
        cmap (str): Colormap for the heatmap
        title (str): Plot title
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Convert to NumPy array if it's a list
    if isinstance(load_matrix, list):
        load_matrix = np.array(load_matrix)
    
    # Create figure and plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        load_matrix, 
        cmap=cmap, 
        annot=False, 
        cbar=True, 
        xticklabels=[f"E{i}" for i in range(num_experts)], 
        yticklabels=[f"L{i}" for i in range(num_layers)]
    )
    
    plt.title(title, fontsize=16)
    plt.xlabel("Expert Index", fontsize=12)
    plt.ylabel("Layer Index", fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()

def plot_pruned_matrix(matrix, prune_indices, num_experts=16, prune_up_to_layer=1, 
                      cmap='YlGnBu', figsize=(12, 10), title="Pruned Matrix Visualization"):
    """
    Plot a matrix with pruned experts highlighted in red.
    
    Args:
        matrix (np.ndarray): Input matrix to visualize (shape: num_layers x num_experts)
        prune_indices (np.ndarray): Array of shape (num_layers, p) where p is the number of 
                                   indices to prune for each layer
        num_experts (int): Total number of experts (default: 16)
        prune_up_to_layer (int): Only apply pruning up to this layer index (0-indexed, default: 1)
        cmap (str): Colormap for non-pruned values
        figsize (tuple): Figure size (width, height)
        title (str): Plot title
    """
    # Convert to numpy array if it's a list
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    
    if isinstance(prune_indices, list):
        prune_indices = np.array(prune_indices)
    
    num_layers = matrix.shape[0]
    
    # Check shapes
    if matrix.shape[1] != num_experts:
        raise ValueError(f"Matrix should have shape (num_layers, {num_experts}), got {matrix.shape}")
        
    if prune_indices.shape[0] != num_layers:
        raise ValueError(f"Prune indices first dimension must match number of layers ({num_layers}), got {prune_indices.shape[0]}")
        
    # Create a mask where True means "keep this expert" (don't prune)
    mask = np.ones((num_layers, num_experts), dtype=bool)
    
    # Apply pruning only up to the specified layer
    for layer_idx in range(min(prune_up_to_layer, num_layers)):
        # For each layer, set the specified indices to False (prune them)
        indices_to_prune = prune_indices[layer_idx]
        # Only use valid indices (those within range of num_experts)
        valid_indices = [idx for idx in indices_to_prune if 0 <= idx < num_experts]
        mask[layer_idx, valid_indices] = False
    
    # Create a figure and axis
    plt.figure(figsize=figsize)
    
    # First plot the base heatmap with the original values
    ax = sns.heatmap(matrix, cmap=cmap, cbar=True)
    
    # Then overlay red for the pruned areas
    # We create a mask array for the overlay
    mask_overlay = np.zeros_like(matrix, dtype=bool)
    mask_overlay[~mask] = True  # Areas where mask is False (pruned experts)
    
    # Plot the overlay with red color
    sns.heatmap(matrix, mask=~mask_overlay, cmap=['red'], cbar=False, ax=ax)
    
    plt.title(title)
    plt.xlabel("Expert Index")
    plt.ylabel("Layer Index")
    plt.tight_layout()
    
    return plt.gcf()

if __name__ == "__main__":
    # Example usage
    matrix = load_matrix()
    if matrix is not None:
        # Create a sample prune indices array - 2 indices to prune per layer
        # Shape: (48, 2) - For each of 48 layers, prune 2 specific expert indices
        sample_prune_indices = np.zeros((48, 2), dtype=int)
        for i in range(48):
            # For example, for each layer i, prune experts (i % 8) and (i % 8 + 8)
            sample_prune_indices[i, 0] = i % 8
            sample_prune_indices[i, 1] = (i % 8) + 8
            sample_prune_indices[i, 1] = (i % 8) + 8
        
        # Plot the matrix with pruning up to layer 10
        fig = plot_pruned_matrix(
            matrix, 
            sample_prune_indices,
            prune_up_to_layer=10,
            title="Matrix with Experts Pruned up to Layer 10"
        )
        plt.savefig("pruned_matrix_visualization.png")
        
        # Create expert load heatmap
        fig2 = plot_expert_load_heatmap(matrix)
        plt.savefig("expert_load_heatmap.png")
        
        plt.show()