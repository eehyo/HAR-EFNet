import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


def compute_mmd_distance(X: np.ndarray, Y: np.ndarray, kernel: str = 'rbf', sigma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions
    
    Args:
        X: First distribution samples [n_samples, n_features]
        Y: Second distribution samples [m_samples, n_features]
        kernel: Kernel type ('linear' or 'rbf')
        sigma: Bandwidth parameter for RBF kernel
        
    Returns:
        MMD distance value
    """
    from sklearn.metrics.pairwise import pairwise_kernels
    
    if kernel == 'linear':
        # Linear kernel: K(x, y) = x^T y
        XX = np.mean(pairwise_kernels(X, X, metric='linear'))
        YY = np.mean(pairwise_kernels(Y, Y, metric='linear'))
        XY = np.mean(pairwise_kernels(X, Y, metric='linear'))
    elif kernel == 'rbf':
        # RBF kernel: K(x, y) = exp(-||x-y||^2 / (2*sigma^2))
        gamma = 1.0 / (2 * sigma**2)
        XX = np.mean(pairwise_kernels(X, X, metric='rbf', gamma=gamma))
        YY = np.mean(pairwise_kernels(Y, Y, metric='rbf', gamma=gamma))
        XY = np.mean(pairwise_kernels(X, Y, metric='rbf', gamma=gamma))
    else:
        raise ValueError(f"Unsupported kernel type: {kernel}")
    
    mmd = XX + YY - 2 * XY
    return max(0, mmd)  # Ensure non-negative


def compute_acc_norm(data: np.ndarray) -> np.ndarray:
    """
    Compute accelerometer norm: sqrt(acc_x² + acc_y² + acc_z²)
    Following the original ColloSSL implementation
    
    Args:
        data: Input data [batch_size, window_size, 3] (x, y, z channels)
        
    Returns:
        Accelerometer norm [batch_size, window_size]
    """
    return np.sqrt(np.sum(np.square(data), axis=2))


def compute_mmd_per_channel(data_x: np.ndarray, data_y: np.ndarray, kernel: str = 'rbf', sigma: float = 1.0) -> float:
    """
    Compute MMD per channel and average
    Following the original ColloSSL implementation
    
    Args:
        data_x: First device data [batch_size, window_size, 3]
        data_y: Second device data [batch_size, window_size, 3]
        kernel: Kernel type for MMD computation
        sigma: Bandwidth parameter for RBF kernel
        
    Returns:
        Average MMD across channels
    """
    mmd_sum = 0.0
    num_channels = data_x.shape[2]
    
    for i in range(num_channels):
        channel_x = data_x[:, :, i]  # [batch_size, window_size]
        channel_y = data_y[:, :, i]  # [batch_size, window_size]
        
        # Flatten for MMD computation
        channel_x_flat = channel_x.reshape(-1, 1)  # [batch_size * window_size, 1]
        channel_y_flat = channel_y.reshape(-1, 1)  # [batch_size * window_size, 1]
        
        mmd_sum += compute_mmd_distance(channel_x_flat, channel_y_flat, kernel, sigma)
    
    return mmd_sum / num_channels


def select_positive_device_mmd(anchor_data: torch.Tensor, candidate_devices_data: Dict[str, torch.Tensor], 
                               device_selection_metric: str = 'mmd_acc_norm', kernel: str = 'rbf', 
                               sigma: float = 1.0) -> Tuple[str, Dict[str, float]]:
    """
    Select positive device based on minimum MMD distance
    Following the original ColloSSL implementation with configurable metrics
    
    Args:
        anchor_data: Anchor device data [batch_size, window_size, 3]
        candidate_devices_data: Dict of candidate device data (all devices D_θ)
        device_selection_metric: Metric type ('mmd_acc_norm', 'mmd_acc_per_channel', 'mmd_full_feature')
        kernel: Kernel type for MMD computation
        sigma: Bandwidth parameter for RBF kernel
        
    Returns:
        Tuple of (positive_device_name, mmd_distances_dict)
    """
    # Convert to numpy for computation
    anchor_np = anchor_data.cpu().numpy()  # [batch_size, window_size, 3]
    
    mmd_distances = {}
    
    for device, device_data in candidate_devices_data.items():
        device_np = device_data.cpu().numpy()  # [batch_size, window_size, 3]
        
        if device_selection_metric == 'mmd_acc_norm':
            # Original ColloSSL: accelerometer norm
            anchor_norm = compute_acc_norm(anchor_np)  # [batch_size, window_size]
            device_norm = compute_acc_norm(device_np)  # [batch_size, window_size]
            
            # Flatten for MMD computation
            anchor_flat = anchor_norm.reshape(-1, 1)  # [batch_size * window_size, 1]
            device_flat = device_norm.reshape(-1, 1)  # [batch_size * window_size, 1]
            
            mmd_dist = compute_mmd_distance(anchor_flat, device_flat, kernel, sigma)
            
        elif device_selection_metric == 'mmd_acc_per_channel':
            # Original ColloSSL: per-channel MMD
            mmd_dist = compute_mmd_per_channel(anchor_np, device_np, kernel, sigma)
            
        elif device_selection_metric == 'mmd_full_feature':
            # Full feature vector MMD (all channels)
            anchor_flat = anchor_np.reshape(-1, anchor_np.shape[-1])  # [batch_size * window_size, 3]
            device_flat = device_np.reshape(-1, device_np.shape[-1])  # [batch_size * window_size, 3]
            
            mmd_dist = compute_mmd_distance(anchor_flat, device_flat, kernel, sigma)
            
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
        strategy: Selection strategy ('closest_only', 'hard_negative', 'closest_pos_all_neg', etc.)
        
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
                                device_selection_strategy: str = 'hard_negative',
                                kernel: str = 'rbf', sigma: float = 1.0) -> Tuple[List[str], List[str], Dict[str, float]]:
    """
    Complete device selection process combining MMD computation and strategy selection
    
    Args:
        anchor_data: Anchor device data [batch_size, window_size, 3]
        candidate_devices_data: Dict of candidate device data
        device_selection_metric: MMD computation metric
        device_selection_strategy: Device selection strategy
        kernel: Kernel type for MMD computation
        sigma: Bandwidth parameter for RBF kernel
        
    Returns:
        Tuple of (positive_devices, negative_devices, mmd_distances)
    """
    # Step 1: Compute MMD distances for all candidate devices
    _, mmd_distances = select_positive_device_mmd(
        anchor_data, candidate_devices_data, 
        device_selection_metric=device_selection_metric,
        kernel=kernel, sigma=sigma
    )
    
    # Step 2: Sort devices by MMD distance (closest first)
    device_order = sorted(mmd_distances.keys(), key=lambda k: mmd_distances[k])
    
    # Step 3: Apply device selection strategy
    positive_devices, negative_devices, updated_distances = device_selection_logic(
        device_order, mmd_distances, strategy=device_selection_strategy
    )
    
    return positive_devices, negative_devices, updated_distances


def compute_negative_weights(mmd_distances: Dict[str, float], positive_device: str) -> Dict[str, float]:
    """
    Compute weights for negative devices based on inverse MMD distances
    Following the original ColloSSL implementation: w_i = 1 / MMD(x*, x_i), normalized by max weight
    
    Args:
        mmd_distances: MMD distances for each device
        positive_device: Name of positive device (still included in negatives)
        
    Returns:
        Dict of normalized negative weights for ALL devices
    """
    negative_weights = {}
    
    # Compute weights for all devices (including positive device for negative sampling)
    for device, mmd_dist in mmd_distances.items():
        # w_i = 1 / MMD(x*, x_i) - 논문 공식 그대로
        weight = 1.0 / max(mmd_dist, 1e-8)  # Avoid division by zero
        negative_weights[device] = weight
    
    # Normalize weights by dividing by maximum weight (기존 구현체 방식)
    max_weight = max(negative_weights.values()) if negative_weights else 1.0
    normalized_weights = {k: v / max_weight for k, v in negative_weights.items()}
    
    return normalized_weights


class MultiViewContrastiveLoss(nn.Module):
    """
    Multi-view Contrastive Loss for ColloSSL
    Implements the exact loss function from the paper with configurable negative sample size
    """
    def __init__(self, temperature: float = 0.1, mmd_kernel: str = 'rbf', 
                 mmd_sigma: float = 1.0, neg_sample_size: int = 1, 
                 device_selection_metric: str = 'mmd_acc_norm',
                 device_selection_strategy: str = 'hard_negative'):
        """
        Initialize Multi-view Contrastive Loss
        
        Args:
            temperature: Temperature parameter for softmax (τ)
            mmd_kernel: Kernel type for MMD computation ('linear' or 'rbf')
            mmd_sigma: Bandwidth parameter for RBF kernel
            neg_sample_size: Number of negative samples per device
            device_selection_metric: Device selection metric type
            device_selection_strategy: Device selection strategy type
        """
        super(MultiViewContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.mmd_kernel = mmd_kernel
        self.mmd_sigma = mmd_sigma
        self.neg_sample_size = neg_sample_size
        self.device_selection_metric = device_selection_metric
        self.device_selection_strategy = device_selection_strategy
    
    def compute_similarity(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings
        
        Args:
            z1: First set of embeddings [batch_size, embedding_dim]
            z2: Second set of embeddings [batch_size, embedding_dim]
            
        Returns:
            Similarity scores [batch_size, batch_size]
        """
        # Normalize embeddings
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
        Compute multi-view contrastive loss with configurable negative sample size
        
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
            device_selection_strategy=self.device_selection_strategy,
            kernel=self.mmd_kernel, sigma=self.mmd_sigma
        )
        
        # For compatibility with existing code, use first positive device
        positive_device = positive_devices[0] if positive_devices else list(sync_data.keys())[0]
        
        # Step 2: Compute negative weights using original implementation approach
        negative_weights = compute_negative_weights(mmd_distances, positive_device)
        
        # Step 3: Prepare positive and negative embeddings
        # Positive: synchronized embeddings from positive device
        positive_embeddings = sync_embeddings[positive_device]  # [batch_size, embedding_dim]
        
        # Step 4: Compute similarities according to paper formula with multiple negatives
        losses = []
        positive_probs = []
        
        for i in range(batch_size):
            anchor_emb = anchor_embeddings[i:i+1]  # [1, embedding_dim]
            
            # Positive similarities: sim(z*, z+_i) for synchronized samples
            pos_emb = positive_embeddings[i:i+1]  # [1, embedding_dim]
            pos_sim = self.compute_similarity(anchor_emb, pos_emb)[0, 0]  # scalar
            pos_sim_scaled = pos_sim / self.temperature
            
            # Negative similarities: sim(z*, z-_j) for multiple asynchronous samples from ALL devices
            neg_sims_scaled = []
            
            for device_name, async_emb in async_embeddings.items():
                device_weight = negative_weights[device_name]
                
                # async_emb: [batch_size, neg_sample_size, embedding_dim]
                device_async_emb = async_emb[i]  # [neg_sample_size, embedding_dim]
                
                # Compute similarities with all negative samples from this device
                neg_sim_matrix = self.compute_similarity(anchor_emb, device_async_emb)  # [1, neg_sample_size]
                
                # Apply device weight to all similarities from this device
                for j in range(self.neg_sample_size):
                    neg_sim = neg_sim_matrix[0, j]  # scalar
                    neg_sim_scaled = (neg_sim / self.temperature) * device_weight
                    neg_sims_scaled.append(neg_sim_scaled)
            
            # Apply multi-view contrastive loss formula
            # L_MCL = -log [Σ exp(pos) / (Σ exp(pos) + Σ w_j * exp(neg))]
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
    Complete ColloSSL loss combining contrastive learning with optional auxiliary losses
    """
    def __init__(self, temperature: float = 0.1, mmd_kernel: str = 'rbf', 
                 mmd_sigma: float = 1.0, neg_sample_size: int = 1, 
                 device_selection_metric: str = 'mmd_acc_norm',
                 device_selection_strategy: str = 'hard_negative'):
        """
        Initialize ColloSSL loss
        
        Args:
            temperature: Temperature parameter for contrastive loss
            mmd_kernel: Kernel type for MMD computation
            mmd_sigma: Bandwidth parameter for RBF kernel
            neg_sample_size: Number of negative samples per device
            device_selection_metric: Device selection metric type
            device_selection_strategy: Device selection strategy type
        """
        super(ColloSSLLoss, self).__init__()
        
        self.contrastive_loss = MultiViewContrastiveLoss(
            temperature=temperature,
            mmd_kernel=mmd_kernel,
            mmd_sigma=mmd_sigma,
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
        Compute complete ColloSSL loss
        
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