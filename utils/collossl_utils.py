import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


# =============================================================================
# MMD (Maximum Mean Discrepancy) Calculation Utilities
# =============================================================================

def pairwise_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the squared pairwise Euclidean distances between x and y.

    Mathematical formula: ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2⟨x_i, y_j⟩
    
    Args:
        x: a tensor of shape [num_x_samples, num_features]
        y: a tensor of shape [num_y_samples, num_features]
    Returns:
        a distance matrix of dimensions [num_x_samples, num_y_samples]
    """
    
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError("Both inputs should be matrices.")
    
    if x.shape[1] != y.shape[1]:
        raise ValueError("The number of features should be the same.")
    
    # Compute squared Euclidean distance matrix
    x_norm = np.sum(np.square(x), axis=1, keepdims=True)  # [num_x_samples, 1]
    y_norm = np.sum(np.square(y), axis=1, keepdims=True)  # [num_y_samples, 1]
    
    # dist = ||x||^2 + ||y||^2 - 2<x,y>
    dist = x_norm + y_norm.T - 2 * np.dot(x, y.T)
    return dist


def gaussian_rbf_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes a Gaussian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Following original ColloSSL TensorFlow implementation.
    
    Mathematical formula: K(x,y) = Σᵢ exp(-||x-y||²/(2σᵢ²))
    
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
    Returns:
        A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel
    """
    # The values usually stays within (-5 ~ 10)
    sigmas = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 100])
    
    beta = 1.0 / (2.0 * sigmas.reshape(-1, 1))  # [num_sigmas, 1]
    dist = pairwise_distance(x, y)  # [num_x_samples, num_y_samples]
    
    # Reshape dist to [1, num_x_samples * num_y_samples]
    dist_flat = dist.reshape(1, -1)
    
    # Compute s = beta * dist_flat: [num_sigmas, num_x_samples * num_y_samples]
    s = np.dot(beta, dist_flat)
    
    # Compute exp(-s) and sum over sigmas
    exp_s = np.exp(-1.0 * s)  # [num_sigmas, num_x_samples * num_y_samples]
    kernel_sum = np.sum(exp_s, axis=0)  # [num_x_samples * num_y_samples]
    
    # Reshape back to original shape
    kernel = kernel_sum.reshape(dist.shape)
    return kernel


def maximum_mean_discrepancy(x: np.ndarray, y: np.ndarray, kernel=gaussian_rbf_kernel) -> float:
    """
    Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Following original ColloSSL TensorFlow implementation
    
    Mathematical formula: MMD²(P, Q) = || E{φ(x)} - E{φ(y)} ||²
                         = E{ K(x, x) } + E{ K(y, y) } - 2 E{ K(x, y) }
    
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss
    """
    # E{ K(x, x) } + E{ K(y, y) } - 2 E{ K(x, y) }
    cost = (
        np.mean(kernel(x, x))      # E{ K(x, x) }
        + np.mean(kernel(y, y))    # E{ K(y, y) }
        - 2 * np.mean(kernel(x, y))  # -2 E{ K(x, y) }
    )
    # We do not allow the loss to become negative
    cost = max(0, cost)
    return cost

# mmd_loss
def compute_mmd_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.
    Using original ColloSSL multi-gaussian RBF kernel implementation.
    
    This is the main MMD computation function used throughout the framework.
    
    Args:
        X: First distribution samples [n_samples, n_features]
        Y: Second distribution samples [m_samples, n_features]
        
    Returns:
        MMD distance value
    """
    # Use original ColloSSL MMD implementation
    mmd_value = maximum_mean_discrepancy(X, Y, kernel=gaussian_rbf_kernel)
    # Apply minimum threshold like original implementation
    mmd_value = max(1e-4, mmd_value)
    return mmd_value




# Accelerometer Data Preprocessing
# compute_acc_norm(data): \sqrt{x^2+y^2+z^2} → [B,T]
def compute_acc_norm(data: np.ndarray) -> np.ndarray:
    """
    Compute accelerometer norm
    Mathematical formula: ||acc|| = √(acc_x² + acc_y² + acc_z²)
    
    Args:
        data: Input accelerometer data [batch_size, window_size, 3] (x, y, z channels)
    Returns:
        Accelerometer norm [batch_size, window_size]
    """
    return np.sqrt(np.sum(np.square(data), axis=2))

# compute_mmd_per_channel(x,y): per channel(flat→BT,1) MMD → average
def compute_mmd_per_channel(data_x: np.ndarray, data_y: np.ndarray) -> float:
    """
    Compute MMD per channel and average
    Following the original ColloSSL implementation
    
    This function computes MMD for each accelerometer channel (x, y, z) separately
    and returns the average, providing a channel-wise distribution comparison.
    
    Args:
        data_x: First device data [batch_size, window_size, 3]
        data_y: Second device data [batch_size, window_size, 3]
    Returns:
        Average MMD across all channels
    """
    mmd_sum = 0.0
    num_channels = data_x.shape[2]
    
    for i in range(num_channels):
        # Extract individual channel data
        channel_x = data_x[:, :, i]  # [batch_size, window_size]
        channel_y = data_y[:, :, i]  # [batch_size, window_size]
        
        # Flatten for MMD computation
        channel_x_flat = channel_x.reshape(-1, 1)  # [batch_size * window_size, 1]
        channel_y_flat = channel_y.reshape(-1, 1)  # [batch_size * window_size, 1]
        
        # Compute MMD for this channel
        mmd_sum += compute_mmd_distance(channel_x_flat, channel_y_flat)
    
    return mmd_sum / num_channels


# =============================================================================
# Device Selection Logic
# =============================================================================

# determining which devices serve as positive and negative samples based on
# MMD distances and configurable selection strategies.

def select_positive_device_mmd(anchor_data: torch.Tensor, candidate_devices_data: Dict[str, torch.Tensor], 
                               device_selection_metric: str = 'mmd_acc_norm') -> Tuple[str, Dict[str, float]]:
    """
    Select positive device based on minimum MMD distance.
    Following the original ColloSSL implementation with configurable metrics.
    
    This function computes MMD distances between anchor device and all candidate
    devices, then selects the device with minimum distance as the positive device.
    
    Args:
        anchor_data: Anchor device data [batch_size, window_size, 3]
        candidate_devices_data: Dict of candidate device data (all devices D_θ)
        device_selection_metric: Metric type for MMD computation:
            - 'mmd_acc_norm': Accelerometer norm-based MMD
            - 'mmd_acc_per_channel': Per-channel MMD average
            - 'mmd_full_feature': Full feature vector MMD
    Returns:
        Tuple of (positive_device_name, mmd_distances_dict)
    """
    anchor_np = anchor_data.cpu().numpy()  # [batch_size, window_size, 3]
    
    mmd_distances = {}
    
    for device, device_data in candidate_devices_data.items():
        device_np = device_data.cpu().numpy()  # [batch_size, window_size, 3]
        
        if device_selection_metric == 'mmd_acc_norm':
            anchor_norm = compute_acc_norm(anchor_np)  # [batch_size, window_size]
            device_norm = compute_acc_norm(device_np)  # [batch_size, window_size]
            
            # Flatten for MMD computation
            anchor_flat = anchor_norm.reshape(-1, 1)  # [batch_size * window_size, 1]
            device_flat = device_norm.reshape(-1, 1)  # [batch_size * window_size, 1]
            
            mmd_dist = compute_mmd_distance(anchor_flat, device_flat)
            
        elif device_selection_metric == 'mmd_acc_per_channel':
            mmd_dist = compute_mmd_per_channel(anchor_np, device_np)
            
        # elif device_selection_metric == 'mmd_full_feature':
        #     anchor_flat = anchor_np.reshape(-1, anchor_np.shape[-1])  # [batch_size * window_size, 3]
        #     device_flat = device_np.reshape(-1, device_np.shape[-1])  # [batch_size * window_size, 3]
            
        #     mmd_dist = compute_mmd_distance(anchor_flat, device_flat)
            
        else:
            raise ValueError(f"Unsupported device selection metric: {device_selection_metric}")
        
        mmd_distances[device] = mmd_dist
    
    # Select device with minimum MMD distance as positive (Closest Positive policy)
    positive_device = min(mmd_distances.keys(), key=lambda k: mmd_distances[k])
    
    return positive_device, mmd_distances


def device_selection_logic(device_order: List[str], pairwise_distances: Dict[str, float], 
                          strategy: str = 'hard_negative') -> Tuple[List[str], List[str], Dict[str, float]]:
    """
    Device selection logic for ColloSSL based on original implementation
    Implements various strategies for selecting positive and negative devices
    
    Args:
        device_order: List of devices ordered by MMD distance (closest first)
        pairwise_distances: Dict of MMD distances for each device
        strategy: Selection strategy for positive/negative device selection:
            - 'closest_only': Only closest device (both positive and negative)
            - 'hard_negative': Closest device as positive, 2nd and 3rd closest as negative
            - 'closest_pos_all_neg': Closest device as positive, all devices as negative
            - 'harder_negative': 4th closest device as positive, 1st and 2nd closest as negative
            - 'closest_pos_rest_neg': Closest device as positive, all others as negative
            - 'closest_two': Closest device as positive, 2nd closest as negative
            - 'closest_two_reverse': 2nd closest device as positive, closest as negative
            - 'random_selection': Random device selection
            - 'mid_selection': Middle devices as positive, farthest devices as negative
            - 'closest_pos_random_neg': Closest device as positive, random device as negative
    Returns:
        Tuple of (positive_devices, negative_devices, updated_pairwise_distances)
    """
    if strategy == 'closest_only':
        # Only closest device (both positive and negative)
        positive_devices = negative_devices = [device_order[0]]
        
    elif strategy == 'closest_pos_all_neg':
        # Closest device as positive, all devices as negative
        positive_devices = [device_order[0]]
        negative_devices = device_order.copy()
        negative_devices.sort()
        
    elif strategy == 'hard_negative':
        # Closest device as positive, 2nd and 3rd closest as negative
        positive_devices = [device_order[0]]
        negative_devices = device_order[1:3].copy()
        negative_devices.sort()
        
    elif strategy == 'harder_negative':
        # 4th closest device as positive, 1st and 2nd closest as negative
        positive_devices = [device_order[3]]
        negative_devices = device_order[0:2].copy()
        negative_devices.sort()
        
    elif strategy == 'closest_pos_rest_neg':
        # Closest device as positive, all others as negative
        positive_devices = [device_order[0]]
        negative_devices = device_order[1:].copy()
        negative_devices.sort()
        
    elif strategy == 'closest_two':
        # Closest device as positive, 2nd closest as negative
        positive_devices = [device_order[0]]
        negative_devices = [device_order[1]]
        
    elif strategy == 'closest_two_reverse':
        # 2nd closest device as positive, closest as negative
        positive_devices = [device_order[1]]
        negative_devices = [device_order[0]]
        
    elif strategy == 'random_selection':
        # Random device selection
        positive_devices = [device_order[np.random.randint(len(device_order))]]
        negative_devices = [device_order[np.random.randint(len(device_order))]]
        # Set all distances to 1.0 for random selection
        pairwise_distances = {key: 1.0 for key in pairwise_distances.keys()}
        
    elif strategy == 'mid_selection':
        # Middle devices as positive, farthest devices as negative
        positive_devices = device_order[1:3].copy()
        negative_devices = device_order[3:].copy()
        
    elif strategy == 'closest_pos_random_neg':
        # Closest device as positive, random device as negative
        positive_devices = [device_order[0]]
        negative_devices = [device_order[np.random.randint(len(device_order))]]
        
    else:
        raise ValueError(f"Unsupported device selection strategy: {strategy}")
    
    return positive_devices, negative_devices, pairwise_distances


def select_devices_with_strategy(anchor_data: torch.Tensor, candidate_devices_data: Dict[str, torch.Tensor],
                                device_selection_metric: str = 'mmd_acc_norm', 
                                device_selection_strategy: str = 'hard_negative') -> Tuple[List[str], List[str], Dict[str, float]]:
    """
    Complete device selection process combining MMD computation and strategy selection
    
    Args:
        anchor_data: Anchor device data [batch_size, window_size, 3]
        candidate_devices_data: Dict of candidate device data
        device_selection_metric: MMD computation metric
        device_selection_strategy: Device selection strategy
    Returns:
        Tuple of (positive_devices, negative_devices, mmd_distances)
    """
    # Step 1: Compute MMD distances for all candidate devices
    _, mmd_distances = select_positive_device_mmd(
        anchor_data, candidate_devices_data, 
        device_selection_metric=device_selection_metric
    )
    
    # Step 2: Sort devices by MMD distance (closest first)
    device_order = sorted(mmd_distances.keys(), key=lambda k: mmd_distances[k])
    
    # Step 3: Apply device selection strategy
    positive_devices, negative_devices, updated_distances = device_selection_logic(
        device_order, mmd_distances, strategy=device_selection_strategy
    )
    
    return positive_devices, negative_devices, updated_distances





# Negative Sample Weighting
# determining the importance of different negative samples based on MMD distances.
def compute_negative_weights(mmd_distances: Dict[str, float], positive_device: str) -> Dict[str, float]:
    """
    Compute weights for negative devices based on inverse MMD distances.
    Following original ColloSSL implementation: w_i = 1 / MMD(x*, x_i).
    
    This function computes raw weights without normalization.
    Normalization is done separately in the forward function.
    
    Mathematical formula: w_i = 1 / MMD(x*, x_i)
    
    Args:
        mmd_distances: MMD distances for each device
        positive_device: Name of positive device (still included in negatives)
    Returns:
        Dict of raw negative weights for ALL devices (not normalized)
    """
    negative_weights = {}
    
    # Compute weights for all devices (including positive device for negative sampling)
    for device, mmd_dist in mmd_distances.items():
        # w_i = 1 / MMD(x*, x_i) - following original implementation exactly
        weight = 1.0 / max(mmd_dist, 1e-8)  # Avoid division by zero
        negative_weights[device] = weight
    
    return negative_weights


# =============================================================================
# Multi-View Contrastive Loss Implementation
# =============================================================================
# 
# These classes implement the core contrastive learning loss functions for ColloSSL,
# combining MMD-based device selection with multi-view contrastive learning.

class MultiViewContrastiveLoss(nn.Module):
    """
    Multi-view Contrastive Loss for ColloSSL.
    Implements the exact loss function from the paper with configurable negative sample size.
    
    This loss function implements the core ColloSSL algorithm:
    1. MMD-based device selection to determine positive/negative devices
    2. Multi-view contrastive learning with weighted negative samples
    3. Temperature-scaled cosine similarity computation
    """
    def __init__(self, temperature: float = 0.1, neg_sample_size: int = 1, 
                 device_selection_metric: str = 'mmd_acc_norm',
                 device_selection_strategy: str = 'hard_negative'):
        """
        Initialize Multi-view Contrastive Loss
        
        Args:
            temperature: Temperature parameter for softmax (τ)
            neg_sample_size: Number of negative samples per device
            device_selection_metric: Device selection metric type for MMD computation
            device_selection_strategy: Device selection strategy for positive/negative selection
        """
        super(MultiViewContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.neg_sample_size = neg_sample_size
        self.device_selection_metric = device_selection_metric
        self.device_selection_strategy = device_selection_strategy
    
    def compute_similarity(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings
        
        Mathematical formula: sim(z1, z2) = z1_norm · z2_norm^T
        where z_norm = z / ||z||_2 (L2 normalization)
        
        Args:
            z1: First set of embeddings [batch_size, embedding_dim]
            z2: Second set of embeddings [batch_size, embedding_dim]
        Returns:
            Similarity scores [batch_size, batch_size]
        """
        # L2 normalize embeddings
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(z1_norm, z2_norm.t())
        return similarity
    
    def forward(self, anchor_embeddings: torch.Tensor, 
                sync_embeddings: Dict[str, torch.Tensor],
                async_embeddings: Dict[str, torch.Tensor],
                anchor_data: torch.Tensor,
                sync_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multi-view contrastive loss with configurable negative sample size.
        
        This function implements the core ColloSSL contrastive learning algorithm:
        1. MMD-based device selection to determine positive/negative devices
        2. Weighted negative sample computation based on MMD distances
        3. Multi-view contrastive loss computation with temperature scaling
        
        Mathematical formula: L_MCL = -log [exp(sim(z*, z+)/τ) / (exp(sim(z*, z+)/τ) + Σ w_j * exp(sim(z*, z-_j)/τ))]
        
        Args:
            anchor_embeddings: Anchor device embeddings [batch_size, embedding_dim]
            sync_embeddings: Dict of synchronized device embeddings (for positive)
            async_embeddings: Dict of asynchronous device embeddings (for negative) 
                            [batch_size, neg_sample_size, embedding_dim] per device
            anchor_data: Raw anchor device data [batch_size, window_size, 3]
            sync_data: Dict of synchronized device data [batch_size, window_size, 3]
            
        Returns:
            Tuple of (loss_value, info_dict)
        """
        batch_size = anchor_embeddings.size(0)
        device = anchor_embeddings.device
        
        # Step 1: MMD-based device selection with strategy
        positive_devices, negative_devices, mmd_distances = select_devices_with_strategy(
            anchor_data, sync_data, 
            device_selection_metric=self.device_selection_metric,
            device_selection_strategy=self.device_selection_strategy
        )
        
        # For compatibility with existing code, use first positive device
        positive_device = positive_devices[0] if positive_devices else list(sync_data.keys())[0]
        
        # Step 2: Prepare positive and negative embeddings
        # Positive: synchronized embeddings from positive device
        positive_embeddings = sync_embeddings[positive_device]  # [batch_size, embedding_dim]
        
        # Step 3: Compute weights
        # Positive weights: all 1.0 (following original)
        positive_weights = torch.ones(batch_size, device=device)
        
        # Negative weights: 1 / MMD distance for each device (following original)
        negative_weights_dict = compute_negative_weights(mmd_distances, positive_device)
        
        # Convert to tensor format matching TensorFlow implementation
        # Create weights for each negative device, then repeat for neg_sample_size
        negative_weights_list = []
        for device_name in async_embeddings.keys():
            device_weight = negative_weights_dict[device_name]
            # Create weight tensor for this device: [batch_size]
            device_weight_tensor = torch.full((batch_size,), device_weight, device=device)
            # Repeat for each negative sample from this device
            for _ in range(self.neg_sample_size):
                negative_weights_list.append(device_weight_tensor)
        
        if negative_weights_list:
            negative_weights = torch.stack(negative_weights_list)  # [num_neg_samples, batch_size]
        else:
            negative_weights = torch.ones(1, batch_size, device=device)
        
        # Step 4: Weight normalization
        # Normalize weights using max across all weights (following TensorFlow implementation)
        max_weight = torch.maximum(torch.max(positive_weights), torch.max(negative_weights))
        positive_weights = positive_weights / max_weight
        negative_weights = negative_weights / max_weight
        
        # Step 5: Compute contrastive loss following TensorFlow implementation
        losses = []
        positive_probs = []
        
        for i in range(batch_size):
            anchor_emb = anchor_embeddings[i:i+1]  # [1, embedding_dim]
            
            # Positive similarity: sim(z*, z+_i) for synchronized samples
            pos_emb = positive_embeddings[i:i+1]  # [1, embedding_dim]
            pos_sim = self.compute_similarity(anchor_emb, pos_emb)[0, 0]  # scalar
            pos_sim_scaled = pos_sim / self.temperature
            
            # Negative similarities: sim(z*, z-_j) for multiple asynchronous samples
            neg_sims_scaled = []
            
            neg_idx = 0
            for device_name, async_emb in async_embeddings.items():
                # async_emb: [batch_size, neg_sample_size, embedding_dim]
                device_async_emb = async_emb[i]  # [neg_sample_size, embedding_dim]
                
                # Compute similarities with all negative samples from this device
                neg_sim_matrix = self.compute_similarity(anchor_emb, device_async_emb)  # [1, neg_sample_size]
                
                # Apply device weight to all similarities from this device
                for j in range(self.neg_sample_size):
                    neg_sim = neg_sim_matrix[0, j]  # scalar
                    neg_weight = negative_weights[neg_idx, i]  # Get weight for this specific negative sample
                    neg_sim_scaled = (neg_sim / self.temperature) * neg_weight
                    neg_sims_scaled.append(neg_sim_scaled)
                    neg_idx += 1
            
            # Apply contrastive loss formula following TensorFlow implementation
            # L = -log [exp(pos) / (exp(pos) + Σ w_j * exp(neg))]
            pos_exp = torch.exp(pos_sim_scaled)
            neg_exp_sum = torch.sum(torch.stack([torch.exp(neg_sim) for neg_sim in neg_sims_scaled]))
            
            # Compute loss for this sample
            denominator = pos_exp + neg_exp_sum
            sample_loss = -torch.log(pos_exp / denominator)
            losses.append(sample_loss)
            
            # Store positive probability for monitoring
            pos_prob = pos_exp / denominator
            positive_probs.append(pos_prob.item())
        
        # Average loss over batch
        total_loss = torch.stack(losses).mean()
        
        # Information for monitoring
        info = {
            'positive_device': positive_device,
            'mmd_distances': mmd_distances,
            'negative_weights': negative_weights,
            'avg_positive_prob': np.mean(positive_probs) if positive_probs else 0.0,
            'num_negative_devices': len(async_embeddings),
            'negative_devices': list(async_embeddings.keys()),
            'total_negative_samples': len(async_embeddings) * self.neg_sample_size,
            'device_selection_metric': self.device_selection_metric
        }
        
        return total_loss, info


class ColloSSLLoss(nn.Module):
    """
    Complete ColloSSL loss combining contrastive learning with optional auxiliary losses.
    
    This is the main loss class that wraps the MultiViewContrastiveLoss and provides
    a unified interface for ColloSSL training. It can be extended to include
    additional auxiliary losses in the future.
    """
    def __init__(self, temperature: float = 0.1, neg_sample_size: int = 1, 
                 device_selection_metric: str = 'mmd_acc_norm',
                 device_selection_strategy: str = 'hard_negative'):
        """
        Initialize ColloSSL loss.
        
        Args:
            temperature: Temperature parameter for contrastive loss (τ)
            neg_sample_size: Number of negative samples per device
            device_selection_metric: Device selection metric type for MMD computation
            device_selection_strategy: Device selection strategy for positive/negative selection
        """
        super(ColloSSLLoss, self).__init__()
        
        # Initialize the core contrastive loss component
        self.contrastive_loss = MultiViewContrastiveLoss(
            temperature=temperature,
            neg_sample_size=neg_sample_size,
            device_selection_metric=device_selection_metric,
            device_selection_strategy=device_selection_strategy
        )
    
    def forward(self, anchor_embeddings: torch.Tensor,
                sync_embeddings: Dict[str, torch.Tensor],
                async_embeddings: Dict[str, torch.Tensor],
                anchor_data: torch.Tensor,
                sync_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute complete ColloSSL loss.
        
        This function computes the total ColloSSL loss by combining the contrastive loss
        with potential auxiliary losses. Currently, only the contrastive loss is used,
        but this provides a framework for future extensions.
        
        Args:
            anchor_embeddings: Anchor device embeddings
            sync_embeddings: Dict of synchronized device embeddings  
            async_embeddings: Dict of asynchronous device embeddings
            anchor_data: Raw anchor device data
            sync_data: Dict of synchronized device data
        Returns:
            Tuple of (total_loss, info_dict)
        """
        # Compute contrastive loss
        contrastive_loss, contrastive_info = self.contrastive_loss(
            anchor_embeddings, sync_embeddings, async_embeddings, anchor_data, sync_data
        )
        
        # Total loss (can be extended with auxiliary losses)
        total_loss = contrastive_loss
        
        # Combine info
        info = {
            'contrastive_loss': contrastive_loss.item(),
            'total_loss': total_loss.item(),
            **contrastive_info
        }
        
        return total_loss, info 
