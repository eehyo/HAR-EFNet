import numpy as np
from typing import List, Callable, Tuple, Union, Any
from dataloaders.simclr_transformations import (
    noise_transform_vectorized,
    scaling_transform_vectorized,
    rotation_transform_vectorized,
    negate_transform_vectorized,
    time_flip_transform_vectorized,
    channel_shuffle_transform_vectorized,
    time_segment_permutation_transform_improved,
    time_warp_transform_low_cost
)

__author__ = "C. I. Tang"
__copyright__ = """Copyright (C) 2020 C. I. Tang"""

"""
This file includes software licensed under the Apache License 2.0, modified by C. I. Tang.

Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
Contact: cit27@cl.cam.ac.uk
License: GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# Type alias for transformation functions
TransformFunction = Callable[[np.ndarray], np.ndarray]


def generate_composite_transform_function_simple(transform_funcs: List[TransformFunction]) -> TransformFunction:
    """
    Create a composite transformation function by composing transformation functions

    Parameters:
        transform_funcs: list of transformation functions
            the function is composed by applying 
            transform_funcs[0] -> transform_funcs[1] -> ...
            i.e. f(x) = f3(f2(f1(x)))

    Returns:
        combined_transform_func: a composite transformation function
    """
    # for i, func in enumerate(transform_funcs):
    #     print(i, func)
    def combined_transform_func(sample: np.ndarray) -> np.ndarray:
        """
        Apply composite transformation to batch data
        
        Args:
            sample: Input batch data [batch_size, window_size, channels]
            
        Returns:
            Transformed batch data [batch_size, window_size, channels]
        """
        for func in transform_funcs:
            sample = func(sample)
        return sample
    
    return combined_transform_func


def get_transform_function_by_name(transform_name: str) -> TransformFunction:
    """
    Get transformation function by name string
    
    Args:
        transform_name: Name of the transformation function
        
    Returns:
        Transformation function
        
    Raises:
        ValueError: If transform name is not supported
    """
    transform_map: dict[str, TransformFunction] = {
        'noise': noise_transform_vectorized,
        'scaling': scaling_transform_vectorized,
        'rotation': rotation_transform_vectorized,
        'negate': negate_transform_vectorized,
        'time_flip': time_flip_transform_vectorized,
        'channel_shuffle': channel_shuffle_transform_vectorized,
        'time_segment_permutation': time_segment_permutation_transform_improved,
        'time_warp': time_warp_transform_low_cost
    }
    
    if transform_name not in transform_map:
        raise ValueError(f"Unsupported transformation: {transform_name}. "
                        f"Available: {list(transform_map.keys())}")
    
    return transform_map[transform_name]


def create_simclr_transformation_function(transform_names: List[str]) -> TransformFunction:
    """
    Create SimCLR transformation function from list of transform names
    
    Args:
        transform_names: List of transformation function names
        
    Returns:
        Composite transformation function
        
    Raises:
        ValueError: If transform_names is empty
    """
    if not transform_names:
        raise ValueError("transform_names cannot be empty")
    
    # Get transformation functions
    transform_funcs: List[TransformFunction] = [get_transform_function_by_name(name) for name in transform_names]
    
    # Create composite function
    transformation_function: TransformFunction = generate_composite_transform_function_simple(transform_funcs)
    
    return transformation_function


def generate_contrastive_views_batch(batch_np: np.ndarray, 
                                   transformation_function: TransformFunction) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two contrastive views from batch data using vectorized transformation
    
    Args:
        batch_np: Input batch data [batch_size, window_size, channels]
        transformation_function: Function to apply transformations
        
    Returns:
        Tuple of (view1, view2) numpy arrays, each with shape [batch_size, window_size, channels]
    """
    # Apply transformation twice to get two different views
    # Stochastic operations in the transformation function will produce different results
    view1_np: np.ndarray = transformation_function(batch_np.copy())
    view2_np: np.ndarray = transformation_function(batch_np.copy())
    
    return view1_np, view2_np 