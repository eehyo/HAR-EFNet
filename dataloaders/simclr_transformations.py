import numpy as np
import math
from scipy.interpolate import CubicSpline
import itertools
from typing import List, Callable


def jitter_transform(X: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Gaussian noise 추가
    Args:
        X: Input data [T, C]
        sigma: Noise standard deviation
    Returns:
        Transformed data [T, C]
    """
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise


def scaling_transform(X: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    채널별 스케일링
    Args:
        X: Input data [T, C]
        sigma: Scaling factor standard deviation
    Returns:
        Transformed data [T, C]
    """
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    return X * scaling_factor


def rotation_transform(X: np.ndarray) -> np.ndarray:
    """
    3D 회전 변환 (센서별로 적용)
    Args:
        X: Input data [T, C] - C는 3의 배수여야 함
    Returns:
        Transformed data [T, C]
    """
    C = X.shape[1]
    if C % 3 != 0:
        raise ValueError(f"Channel dimension ({C}) must be multiple of 3 for proper sensor rotation")
    
    n_sensors = C // 3
    X_rot = np.zeros_like(X)

    for i in range(n_sensors):
        axis = np.random.uniform(low=-1, high=1, size=3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        R = axis_angle_to_rotation_matrix(axis, angle)

        vec = X[:, i*3:(i+1)*3]  # [T, 3]
        X_rot[:, i*3:(i+1)*3] = vec @ R

    return X_rot


def axis_angle_to_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    축-각도로부터 회전행렬 생성
    """
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    
    return np.array([
        [x*xC+c, xyC-zs, zxC+ys],
        [xyC+zs, y*yC+c, yzC-xs],
        [zxC-ys, yzC+xs, z*zC+c]
    ])


def negate_transform(X: np.ndarray) -> np.ndarray:
    """
    신호 반전
    Args:
        X: Input data [T, C]
    Returns:
        Transformed data [T, C]
    """
    return -X


def horizontal_flip_transform(X: np.ndarray) -> np.ndarray:
    """
    시간 축 반전
    Args:
        X: Input data [T, C]
    Returns:
        Transformed data [T, C]
    """
    return np.flip(X, axis=0)


def channel_shuffle_transform(X: np.ndarray) -> np.ndarray:
    """
    채널 순서 셔플
    Args:
        X: Input data [T, C]
    Returns:
        Transformed data [T, C]
    """
    C = X.shape[1]
    perm = np.random.permutation(C)
    return X[:, perm]


def permutation_transform(X: np.ndarray, nPerm: int = 4, minSegLength: int = 10) -> np.ndarray:
    """
    시간 구간 순열 변환
    Args:
        X: Input data [T, C]
        nPerm: Number of segments
        minSegLength: Minimum segment length
    Returns:
        Transformed data [T, C]
    """
    X_new = np.zeros_like(X)
    while True:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[:-1]) > minSegLength:
            break
    
    idx = np.random.permutation(nPerm)
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
        X_new[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    
    return X_new


def time_warp_transform(X: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    """
    시간 워핑 변환
    Args:
        X: Input data [T, C]
        sigma: Warping strength
        knot: Number of knots for cubic spline
    Returns:
        Transformed data [T, C]
    """
    xx = np.linspace(0, X.shape[0] - 1, num=knot + 2)
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    
    tt_cum = np.zeros_like(X)
    for i in range(X.shape[1]):
        cs = CubicSpline(xx, yy[:, i])
        warped = cs(x_range)
        tt_cum[:, i] = np.cumsum(warped)
        t_scale = (X.shape[0] - 1) / tt_cum[-1, i]
        tt_cum[:, i] *= t_scale
    
    X_new = np.zeros_like(X)
    for i in range(X.shape[1]):
        X_new[:, i] = np.interp(x_range, tt_cum[:, i], X[:, i])
    
    return X_new


def get_simclr_transforms() -> List[Callable]:
    """
    SimCLR을 위한 변환 함수 리스트 반환
    
    Returns:
        List of transformation functions
    """
    transforms = [
        lambda x: jitter_transform(x, sigma=0.05),
        lambda x: scaling_transform(x, sigma=0.1),
        rotation_transform,
        negate_transform,
        horizontal_flip_transform,
        lambda x: permutation_transform(x, nPerm=4, minSegLength=10),
        lambda x: time_warp_transform(x, sigma=0.2, knot=4),
        channel_shuffle_transform
    ]
    
    return transforms


def apply_random_transform_pair(X: np.ndarray) -> tuple:
    """
    두 개의 서로 다른 랜덤 변환을 적용하여 positive pair 생성
    
    Args:
        X: Input data [T, C]
        
    Returns:
        Tuple of (transformed_x1, transformed_x2)
    """
    transforms = get_simclr_transforms()
    
    # 두 개의 서로 다른 변환을 랜덤하게 선택
    selected_transforms = np.random.choice(len(transforms), size=2, replace=False)
    
    transform1 = transforms[selected_transforms[0]]
    transform2 = transforms[selected_transforms[1]]
    
    x1 = transform1(X.copy())
    x2 = transform2(X.copy())
    
    return x1, x2


def apply_transform_combinations(X: np.ndarray) -> List[tuple]:
    """
    모든 변환 조합으로 positive pairs 생성 (validation용)
    
    Args:
        X: Input data [T, C]
        
    Returns:
        List of transformation pairs
    """
    transforms = get_simclr_transforms()
    pairs = []
    
    # 모든 변환 조합 생성
    for i in range(len(transforms)):
        for j in range(i+1, len(transforms)):
            x1 = transforms[i](X.copy())
            x2 = transforms[j](X.copy())
            pairs.append((x1, x2))
    
    return pairs
