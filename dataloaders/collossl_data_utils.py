import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import random

class ColloSSLDataset(Dataset):
    """
    ColloSSL multi-device dataset for contrastive learning
    Treats each IMU sensor (wrist, chest, ankle) as a separate device
    
    Key changes based on paper:
    1. All devices (including positive) are used as negatives
    2. True time synchronization for positive/negative sampling
    3. Batch-level asynchronous sampling for negatives
    """
    def __init__(self, original_dataset, device_channel_mapping: Dict[str, List[int]], 
                 anchor_device: str = 'wrist', batch_size: int = 256):
        """
        Initialize ColloSSL dataset
        
        Args:
            original_dataset: Base PAMAP2 dataset
            device_channel_mapping: Mapping from device names to channel indices
            anchor_device: Primary device for anchor samples
            batch_size: Batch size for asynchronous negative sampling
        """
        self.original_dataset = original_dataset
        self.device_channel_mapping = device_channel_mapping
        self.anchor_device = anchor_device
        self.batch_size = batch_size
        
        # Available devices (all devices are potential positives/negatives)
        self.devices = list(device_channel_mapping.keys())
        self.num_devices = len(self.devices)
        
        # Get anchor device channel indices
        self.anchor_channels = device_channel_mapping[anchor_device]
        
        # Other devices (candidate devices D_θ in paper)
        self.candidate_devices = [d for d in self.devices if d != self.anchor_device]
        
        # Build time-aligned batches for proper synchronization
        self._build_time_aligned_batches()
    
    def _build_time_aligned_batches(self):
        """Build time-aligned batches for synchronous positive and asynchronous negative sampling"""
        self.time_aligned_batches = []
        
        # Group samples into batches
        for start_idx in range(0, len(self.original_dataset), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.original_dataset))
            batch_indices = list(range(start_idx, end_idx))
            self.time_aligned_batches.append(batch_indices)
    
    def extract_device_data(self, x: np.ndarray, device: str) -> np.ndarray:
        """
        Extract data for specific device from full sensor data
        
        Args:
            x: Full sensor data [window_size, num_channels]
            device: Device name ('wrist', 'chest', 'ankle')
            
        Returns:
            Device-specific data [window_size, 3] (x, y, z channels)
        """
        channels = self.device_channel_mapping[device]
        return x[:, channels]
    
    def get_batch_for_index(self, index: int) -> List[int]:
        """Get the batch that contains the given index"""
        for batch in self.time_aligned_batches:
            if index in batch:
                return batch
        return [index]  # Fallback
    
    def get_synchronized_samples(self, anchor_idx: int) -> Dict[str, np.ndarray]:
        """
        Get time-synchronized samples from all candidate devices
        
        Args:
            anchor_idx: Index of anchor sample
            
        Returns:
            Dict of synchronized device data for all candidate devices
        """
        x, _ = self.original_dataset[anchor_idx]
        
        sync_samples = {}
        for device in self.candidate_devices:
            device_data = self.extract_device_data(x, device)
            sync_samples[device] = device_data
        
        return sync_samples
    
    def get_asynchronous_samples(self, anchor_idx: int) -> Dict[str, np.ndarray]:
        """
        Get asynchronous (different time) samples from all candidate devices
        
        Args:
            anchor_idx: Index of anchor sample
            
        Returns:
            Dict of asynchronous device data for all candidate devices
        """
        # Get the batch containing anchor
        anchor_batch = self.get_batch_for_index(anchor_idx)
        
        async_samples = {}
        for device in self.candidate_devices:
            # Sample different time step t' ≠ t from the same batch
            available_indices = [idx for idx in anchor_batch if idx != anchor_idx]
            
            if available_indices:
                async_idx = random.choice(available_indices)
            else:
                # Fallback: sample from a different batch
                other_batches = [b for b in self.time_aligned_batches if anchor_idx not in b]
                if other_batches:
                    other_batch = random.choice(other_batches)
                    async_idx = random.choice(other_batch)
                else:
                    async_idx = anchor_idx  # Last resort
            
            x, _ = self.original_dataset[async_idx]
            device_data = self.extract_device_data(x, device)
            async_samples[device] = device_data
        
        return async_samples
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        """
        Get ColloSSL sample for contrastive learning
        
        Returns:
            Dict containing:
            - anchor_data: Anchor device data [window_size, 3]
            - sync_samples: Dict of synchronized samples from all candidate devices
            - async_samples: Dict of asynchronous samples from all candidate devices
            - anchor_label: Label for anchor sample
            - sample_index: Original sample index
        """
        # Get anchor sample
        x, y = self.original_dataset[index]
        anchor_data = self.extract_device_data(x, self.anchor_device)
        
        # Get synchronized samples (for positive sampling)
        sync_samples = self.get_synchronized_samples(index)
        
        # Get asynchronous samples (for negative sampling)
        async_samples = self.get_asynchronous_samples(index)
        
        return {
            'anchor_data': anchor_data.astype(np.float32),
            'sync_samples': {k: v.astype(np.float32) for k, v in sync_samples.items()},
            'async_samples': {k: v.astype(np.float32) for k, v in async_samples.items()},
            'anchor_label': y,
            'sample_index': index
        }


# Import MMD and device selection functions from utils
from utils.collossl_utils import compute_mmd_distance, select_positive_device_mmd, compute_negative_weights


def create_collossl_dataloader(base_dataloader: DataLoader, device_channel_mapping: Dict[str, List[int]], 
                              anchor_device: str = 'wrist', batch_size: int = 256) -> DataLoader:
    """
    Create ColloSSL dataloader from base PAMAP2 dataloader
    
    Args:
        base_dataloader: Original PAMAP2 dataloader
        device_channel_mapping: Mapping from device names to channel indices
        anchor_device: Anchor device name
        batch_size: Batch size (default from config)
        
    Returns:
        ColloSSL dataloader
    """
    base_dataset = base_dataloader.dataset
    
    # Create ColloSSL dataset
    collossl_dataset = ColloSSLDataset(
        original_dataset=base_dataset,
        device_channel_mapping=device_channel_mapping,
        anchor_device=anchor_device,
        batch_size=batch_size
    )
    
    # Create new dataloader with same settings
    collossl_dataloader = DataLoader(
        dataset=collossl_dataset,
        batch_size=batch_size,  # Use config batch size
        shuffle=getattr(base_dataloader, 'shuffle', True),
        num_workers=getattr(base_dataloader, 'num_workers', 0),
        drop_last=getattr(base_dataloader, 'drop_last', False),
        collate_fn=collossl_collate_fn
    )
    
    return collossl_dataloader


def collossl_collate_fn(batch):
    """
    Custom collate function for ColloSSL batch processing
    
    Args:
        batch: List of sample dicts from ColloSSLDataset
        
    Returns:
        Batched data dict
    """
    batch_size = len(batch)
    
    # Extract anchor data
    anchor_data = torch.stack([torch.tensor(sample['anchor_data']) for sample in batch])
    anchor_labels = torch.tensor([sample['anchor_label'] for sample in batch])
    sample_indices = [sample['sample_index'] for sample in batch]
    
    # Extract synchronized samples (for positive sampling)
    devices = list(batch[0]['sync_samples'].keys())
    sync_samples = {}
    async_samples = {}
    
    for device in devices:
        sync_data_list = []
        async_data_list = []
        
        for sample in batch:
            sync_data_list.append(torch.tensor(sample['sync_samples'][device]))
            async_data_list.append(torch.tensor(sample['async_samples'][device]))
        
        sync_samples[device] = torch.stack(sync_data_list)
        async_samples[device] = torch.stack(async_data_list)
    
    return {
        'anchor_data': anchor_data,
        'sync_samples': sync_samples,
        'async_samples': async_samples,
        'anchor_labels': anchor_labels,
        'sample_indices': sample_indices
    } 