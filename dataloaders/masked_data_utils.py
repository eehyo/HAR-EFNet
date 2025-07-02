import torch
import numpy as np
from torch.utils.data import Dataset


class MaskedReconstructionDataset(Dataset):
    """
    Dataset wrapper for masked reconstruction training
    Applies BERT-style masking to time series data
    """
    def __init__(self, original_dataset, mask_choice):
        super(MaskedReconstructionDataset, self).__init__()
        self.original_dataset = original_dataset
        self.mask_choice = mask_choice
    
    @staticmethod
    def mask_lm(x, choice=0.10):
        """
        Apply BERT-style masking to time series data
        
        Args:
            x: Input tensor [window_size, input_channels]
            choice: Probability of masking each time step
            
        Returns:
            choice_mask: Binary mask indicating which positions were chosen for reconstruction
            x_perturbed: Masked input where some positions are zeroed, replaced, or kept unchanged
        """
        # https://github.com/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT.ipynb
        # https://github.com/huanghonggit/Mask-Language-Model/blob/master/dataset/dataset.py#L78
        # BERT: In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random
        w, c = x.shape
        # mask == 1 : zero masking
        cand = torch.bernoulli(choice * torch.ones((w, 1), device=x.device))            # choice_mask: 1=positions to reconstruct, 0=positions to ignore
        mask = torch.bernoulli(0.8 * torch.ones((w, 1), device=x.device))               # 80% -> zero masked (cand*mask)
        replace_mask = torch.bernoulli(0.5 * torch.ones((w, 1), device=x.device))       # 10% = 50% * 20% unchanged (cand*(1-mask)*keep_mask)
        keep_mask = 1 - replace_mask                                                    # 10% = 50% * 20% perturbed (cand*(1-mask)*replace_mask)

        perturbed_x = x[torch.empty(w).random_(0, w).long(), :]                         # replacement data from random timesteps
        x_perturbed = (1 - cand) * x + cand * (1 - mask) * (perturbed_x * replace_mask + x * keep_mask)   # apply masking strategy
        
        return cand, x_perturbed  # cand: binary mask (1=positions to reconstruct, 0=positions to ignore), x_perturbed: input with masking applied (zeros/random/unchanged at masked positions)
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        # Get original data (x, y) from base dataset
        x, y = self.original_dataset[index]
        
        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Apply masking - use original data as target for reconstruction
        choice_mask, x_perturbed = self.mask_lm(x, choice=self.mask_choice)
        
        # Return (perturbed_input, original_target, reconstruction_mask)
        # x_perturbed: masked input data for model training
        # x: original target data for loss calculation  
        # choice_mask: binary mask indicating which positions were masked (1=masked, 0=unmasked)
        return x_perturbed, x, choice_mask


def create_masked_dataloader(base_dataloader, mask_choice):
    """
    Create a masked reconstruction dataloader from a base dataloader
    
    Args:
        base_dataloader: Original dataloader
        mask_choice: Probability of masking each time step
        
    Returns:
        New dataloader with masked reconstruction dataset
    """
    # Extract dataset from base dataloader
    base_dataset = base_dataloader.dataset
    
    # Create masked dataset
    masked_dataset = MaskedReconstructionDataset(base_dataset, mask_choice)
    
    # Create new dataloader with same settings but masked dataset
    masked_dataloader = torch.utils.data.DataLoader(
        dataset=masked_dataset,
        batch_size=base_dataloader.batch_size,
        shuffle=base_dataloader.dataset.shuffle if hasattr(base_dataloader.dataset, 'shuffle') else True,
        num_workers=getattr(base_dataloader, 'num_workers', 0),
        drop_last=getattr(base_dataloader, 'drop_last', False)
    )
    
    return masked_dataloader 