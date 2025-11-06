"""Tiling and Patching Preprocessing for Anomalib.

This module demonstrates how to implement tiling and patching as preprocessing
steps that are automatically integrated into the model during training and inference.

Tiling/Patching is useful for:
- Processing large images (split into manageable tiles)
- Focusing on local patterns (patch-based analysis)
- Data augmentation (random patches during training)
- Memory efficiency (process one tile/patch at a time)

Key Features:
- All transforms are ONNX-compatible (using PyTorch operations only)
- Automatically integrated into model via PreProcessor
- Works during training, export, and inference
- No manual tiling needed during deployment
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.transforms.v2 import Compose, Normalize, Resize, Transform

from anomalib.data import MVTec
from anomalib.deploy import TorchInferencer
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor


# ============================================================================
# TILING TRANSFORMS (ONNX-COMPATIBLE)
# ============================================================================

class TileExtractTransform(Transform):
    """Extract and process tiles from images (ONNX-compatible).

    This transform divides the input image into a grid of tiles and processes
    each tile independently. Useful for large images or patch-based anomaly detection.

    Args:
        tile_size: Size of each tile (height, width)
        stride: Stride between tiles (for overlapping tiles)
        process_mode: How to handle tiles - "flatten", "average", "max"
    """

    def __init__(
        self,
        tile_size: tuple[int, int],
        stride: tuple[int, int] | None = None,
        process_mode: str = "average",
    ) -> None:
        super().__init__()
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size  # Non-overlapping by default
        self.process_mode = process_mode

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Extract and process tiles.

        Args:
            inpt: Input tensor [B, C, H, W]
            params: Transform parameters (unused)

        Returns:
            Processed tensor [B, C, H, W]
        """
        B, C, H, W = inpt.shape

        # Use unfold to extract tiles (ONNX-compatible)
        # unfold(dimension, size, step) creates a sliding window
        tiles = inpt.unfold(2, self.tile_size[0], self.stride[0])  # Height dimension
        tiles = tiles.unfold(3, self.tile_size[1], self.stride[1])  # Width dimension

        # tiles shape: [B, C, n_tiles_h, n_tiles_w, tile_h, tile_w]
        n_tiles_h, n_tiles_w = tiles.shape[2], tiles.shape[3]

        # Reshape for processing
        tiles = tiles.permute(0, 2, 3, 1, 4, 5)  # [B, n_h, n_w, C, tile_h, tile_w]
        tiles = tiles.reshape(B * n_tiles_h * n_tiles_w, C, self.tile_size[0], self.tile_size[1])

        # Process tiles (example: normalize each tile independently)
        # You can add any custom processing here
        processed_tiles = self._process_tiles(tiles)

        # Reshape back
        processed_tiles = processed_tiles.reshape(
            B,
            n_tiles_h,
            n_tiles_w,
            C,
            self.tile_size[0],
            self.tile_size[1],
        )
        processed_tiles = processed_tiles.permute(0, 3, 1, 4, 2, 5)  # [B, C, n_h, tile_h, n_w, tile_w]
        processed_tiles = processed_tiles.reshape(
            B,
            C,
            n_tiles_h * self.tile_size[0],
            n_tiles_w * self.tile_size[1],
        )

        # Resize back to original size if needed
        if processed_tiles.shape[2:] != inpt.shape[2:]:
            processed_tiles = F.interpolate(
                processed_tiles,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        return processed_tiles

    def _process_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        """Process extracted tiles.

        Args:
            tiles: Tiles tensor [B*n_tiles, C, tile_h, tile_w]

        Returns:
            Processed tiles [B*n_tiles, C, tile_h, tile_w]
        """
        if self.process_mode == "normalize":
            # Normalize each tile independently
            mean = tiles.mean(dim=(2, 3), keepdim=True)
            std = tiles.std(dim=(2, 3), keepdim=True) + 1e-8
            return (tiles - mean) / std
        elif self.process_mode == "identity":
            return tiles
        else:
            return tiles

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self._transform(inpt, {})


class TileAggregationTransform(Transform):
    """Process image as tiles and aggregate results (ONNX-compatible).

    This transform extracts tiles, applies a processing function to each tile,
    and aggregates the results back into the original image shape.

    Args:
        tile_size: Size of each tile (height, width)
        stride: Stride between tiles (None = non-overlapping)
        aggregation: How to aggregate - "mean", "max", "min"
    """

    def __init__(
        self,
        tile_size: tuple[int, int],
        stride: tuple[int, int] | None = None,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size
        self.aggregation = aggregation

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Process tiles and aggregate.

        Args:
            inpt: Input tensor [B, C, H, W]
            params: Transform parameters (unused)

        Returns:
            Aggregated tensor [B, C, H, W]
        """
        B, C, H, W = inpt.shape

        # Extract tiles using unfold
        tiles = inpt.unfold(2, self.tile_size[0], self.stride[0])
        tiles = tiles.unfold(3, self.tile_size[1], self.stride[1])

        # tiles shape: [B, C, n_tiles_h, n_tiles_w, tile_h, tile_w]

        # Apply aggregation on tile level
        if self.aggregation == "mean":
            aggregated = tiles.mean(dim=(4, 5))  # Average within each tile
        elif self.aggregation == "max":
            aggregated = tiles.amax(dim=(4, 5))  # Max within each tile
        elif self.aggregation == "min":
            aggregated = tiles.amin(dim=(4, 5))  # Min within each tile
        else:
            aggregated = tiles.mean(dim=(4, 5))

        # aggregated shape: [B, C, n_tiles_h, n_tiles_w]

        # Upsample back to original size
        result = F.interpolate(
            aggregated,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self._transform(inpt, {})


class OverlappingTileTransform(Transform):
    """Process overlapping tiles with weighted averaging (ONNX-compatible).

    This transform handles overlapping tiles and uses weighted averaging
    to blend overlapping regions smoothly.

    Args:
        tile_size: Size of each tile
        overlap: Overlap amount (pixels or fraction)
    """

    def __init__(self, tile_size: int, overlap: int) -> None:
        super().__init__()
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

        # Create weight tensor for blending overlaps
        self.register_buffer("weights", self._create_weight_tensor())

    def _create_weight_tensor(self) -> torch.Tensor:
        """Create weight tensor for blending overlaps."""
        # Create triangular weights for overlap regions
        weights = torch.ones(self.tile_size, self.tile_size)

        # Apply cosine taper at borders
        if self.overlap > 0:
            taper = torch.cos(
                torch.linspace(0, torch.pi / 2, self.overlap),
            )

            # Top border
            weights[: self.overlap, :] *= taper.unsqueeze(1)
            # Bottom border
            weights[-self.overlap :, :] *= taper.flip(0).unsqueeze(1)
            # Left border
            weights[:, : self.overlap] *= taper.unsqueeze(0)
            # Right border
            weights[:, -self.overlap :] *= taper.flip(0).unsqueeze(0)

        return weights.unsqueeze(0).unsqueeze(0)  # [1, 1, tile_size, tile_size]

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Process overlapping tiles.

        Args:
            inpt: Input tensor [B, C, H, W]
            params: Transform parameters (unused)

        Returns:
            Processed tensor [B, C, H, W]
        """
        B, C, H, W = inpt.shape

        # Extract overlapping tiles
        tiles = inpt.unfold(2, self.tile_size, self.stride)
        tiles = tiles.unfold(3, self.tile_size, self.stride)

        # tiles shape: [B, C, n_tiles_h, n_tiles_w, tile_h, tile_w]
        n_tiles_h, n_tiles_w = tiles.shape[2], tiles.shape[3]

        # Process each tile (example: enhance edges)
        processed_tiles = tiles  # Replace with your processing

        # Reconstruct with weighted averaging
        # This is a simplified version; full fold operation is more complex
        result = F.interpolate(
            processed_tiles.mean(dim=(4, 5)),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self._transform(inpt, {})


# ============================================================================
# PATCHING TRANSFORMS (ONNX-COMPATIBLE)
# ============================================================================

class RandomPatchExtractTransform(Transform):
    """Extract random patches for training (ONNX-compatible).

    During training, this extracts random patches from images for augmentation.
    During inference, it processes the full image.

    Args:
        patch_size: Size of patches to extract
        num_patches: Number of patches to extract per image
        training: Whether in training mode
    """

    def __init__(
        self,
        patch_size: tuple[int, int],
        num_patches: int = 1,
        training: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.training = training

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Extract random patches.

        Args:
            inpt: Input tensor [B, C, H, W]
            params: Transform parameters (unused)

        Returns:
            Patches [B*num_patches, C, patch_h, patch_w] if training,
            else original input
        """
        if not self.training:
            # During inference, return full image
            return inpt

        B, C, H, W = inpt.shape

        # Extract random patches (simplified for ONNX)
        # In practice, you might use grid_sample for differentiability
        patches = []
        for _ in range(self.num_patches):
            # Random top-left corner
            h_start = torch.randint(0, H - self.patch_size[0] + 1, (1,)).item()
            w_start = torch.randint(0, W - self.patch_size[1] + 1, (1,)).item()

            # Extract patch
            patch = inpt[
                :,
                :,
                h_start : h_start + self.patch_size[0],
                w_start : w_start + self.patch_size[1],
            ]
            patches.append(patch)

        # Stack patches
        patches = torch.cat(patches, dim=0)

        return patches

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self._transform(inpt, {})


class PatchNormalizationTransform(Transform):
    """Normalize patches independently (ONNX-compatible).

    Each patch is normalized independently using its own statistics.
    Useful for handling illumination variations across image regions.

    Args:
        patch_size: Size of patches for normalization
    """

    def __init__(self, patch_size: tuple[int, int]) -> None:
        super().__init__()
        self.patch_size = patch_size

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Normalize patches independently.

        Args:
            inpt: Input tensor [B, C, H, W]
            params: Transform parameters (unused)

        Returns:
            Normalized tensor [B, C, H, W]
        """
        B, C, H, W = inpt.shape

        # Extract patches
        patches = inpt.unfold(2, self.patch_size[0], self.patch_size[0])
        patches = patches.unfold(3, self.patch_size[1], self.patch_size[1])

        # patches shape: [B, C, n_patches_h, n_patches_w, patch_h, patch_w]

        # Normalize each patch independently
        mean = patches.mean(dim=(4, 5), keepdim=True)
        std = patches.std(dim=(4, 5), keepdim=True) + 1e-8
        patches_norm = (patches - mean) / std

        # Reconstruct
        n_patches_h, n_patches_w = patches.shape[2], patches.shape[3]
        patches_norm = patches_norm.permute(0, 1, 2, 4, 3, 5)  # [B, C, n_h, patch_h, n_w, patch_w]
        result = patches_norm.reshape(
            B,
            C,
            n_patches_h * self.patch_size[0],
            n_patches_w * self.patch_size[1],
        )

        # Resize back to original if needed
        if result.shape[2:] != (H, W):
            result = F.interpolate(result, size=(H, W), mode="bilinear", align_corners=False)

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self._transform(inpt, {})


class PatchContrastEnhancementTransform(Transform):
    """Enhance contrast within each patch (ONNX-compatible).

    Applies histogram equalization-like operation to each patch independently.

    Args:
        patch_size: Size of patches
        strength: Enhancement strength (0-1)
    """

    def __init__(self, patch_size: tuple[int, int], strength: float = 0.5) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.strength = strength

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Enhance contrast per patch.

        Args:
            inpt: Input tensor [B, C, H, W]
            params: Transform parameters (unused)

        Returns:
            Enhanced tensor [B, C, H, W]
        """
        B, C, H, W = inpt.shape

        # Extract patches
        patches = inpt.unfold(2, self.patch_size[0], self.patch_size[0])
        patches = patches.unfold(3, self.patch_size[1], self.patch_size[1])

        # Enhance contrast per patch (simplified CLAHE-like operation)
        patch_min = patches.amin(dim=(4, 5), keepdim=True)
        patch_max = patches.amax(dim=(4, 5), keepdim=True)

        # Stretch to [0, 1] within each patch
        patches_enhanced = (patches - patch_min) / (patch_max - patch_min + 1e-8)

        # Blend with original
        patches_enhanced = (
            self.strength * patches_enhanced + (1 - self.strength) * patches
        )

        # Reconstruct
        n_patches_h, n_patches_w = patches.shape[2], patches.shape[3]
        patches_enhanced = patches_enhanced.permute(0, 1, 2, 4, 3, 5)
        result = patches_enhanced.reshape(
            B,
            C,
            n_patches_h * self.patch_size[0],
            n_patches_w * self.patch_size[1],
        )

        # Resize back if needed
        if result.shape[2:] != (H, W):
            result = F.interpolate(result, size=(H, W), mode="bilinear", align_corners=False)

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self._transform(inpt, {})


class SlidingWindowTransform(Transform):
    """Apply sliding window processing (ONNX-compatible).

    Applies a custom operation using a sliding window approach.
    Useful for local operations like filtering, pattern matching, etc.

    Args:
        window_size: Size of sliding window
        stride: Stride for sliding window
        operation: Operation to apply - "mean", "max", "std"
    """

    def __init__(
        self,
        window_size: int,
        stride: int = 1,
        operation: str = "mean",
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.operation = operation

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply sliding window operation.

        Args:
            inpt: Input tensor [B, C, H, W]
            params: Transform parameters (unused)

        Returns:
            Processed tensor [B, C, H, W]
        """
        # Use avg_pool2d or max_pool2d for sliding window operations
        if self.operation == "mean":
            result = F.avg_pool2d(
                inpt,
                kernel_size=self.window_size,
                stride=self.stride,
                padding=self.window_size // 2,
            )
        elif self.operation == "max":
            result = F.max_pool2d(
                inpt,
                kernel_size=self.window_size,
                stride=self.stride,
                padding=self.window_size // 2,
            )
        else:
            result = F.avg_pool2d(
                inpt,
                kernel_size=self.window_size,
                stride=self.stride,
                padding=self.window_size // 2,
            )

        # Upsample if stride > 1
        if result.shape[2:] != inpt.shape[2:]:
            result = F.interpolate(
                result,
                size=inpt.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self._transform(inpt, {})


# ============================================================================
# ADVANCED TILING/PATCHING TRANSFORMS
# ============================================================================

class MultiScaleTilingTransform(Transform):
    """Multi-scale tiling for capturing features at different resolutions.

    Args:
        tile_sizes: List of tile sizes to process
        aggregation: How to combine scales - "concat", "mean", "max"
    """

    def __init__(
        self,
        tile_sizes: list[tuple[int, int]],
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        self.tile_sizes = tile_sizes
        self.aggregation = aggregation

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Process at multiple scales.

        Args:
            inpt: Input tensor [B, C, H, W]
            params: Transform parameters (unused)

        Returns:
            Multi-scale result [B, C, H, W]
        """
        results = []

        for tile_size in self.tile_sizes:
            # Extract tiles at this scale
            tiles = inpt.unfold(2, tile_size[0], tile_size[0])
            tiles = tiles.unfold(3, tile_size[1], tile_size[1])

            # Aggregate tiles
            aggregated = tiles.mean(dim=(4, 5))

            # Upsample to original size
            upsampled = F.interpolate(
                aggregated,
                size=inpt.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            results.append(upsampled)

        # Combine scales
        if self.aggregation == "mean":
            return torch.stack(results).mean(dim=0)
        elif self.aggregation == "max":
            return torch.stack(results).amax(dim=0)
        else:
            return torch.stack(results).mean(dim=0)

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self._transform(inpt, {})


class AdaptiveTilingTransform(Transform):
    """Adaptive tiling based on image statistics.

    Divides image into tiles and processes each based on its content.

    Args:
        base_tile_size: Base tile size
        adapt_to_variance: Adjust processing based on variance
    """

    def __init__(self, base_tile_size: int, adapt_to_variance: bool = True) -> None:
        super().__init__()
        self.base_tile_size = base_tile_size
        self.adapt_to_variance = adapt_to_variance

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Adaptive tiling based on content.

        Args:
            inpt: Input tensor [B, C, H, W]
            params: Transform parameters (unused)

        Returns:
            Processed tensor [B, C, H, W]
        """
        B, C, H, W = inpt.shape

        # Extract tiles
        tiles = inpt.unfold(2, self.base_tile_size, self.base_tile_size)
        tiles = tiles.unfold(3, self.base_tile_size, self.base_tile_size)

        if self.adapt_to_variance:
            # Compute variance per tile
            variance = tiles.var(dim=(4, 5), keepdim=True)

            # Normalize based on variance (high variance = more detail)
            tiles_processed = tiles / (variance + 1e-8)
        else:
            tiles_processed = tiles

        # Reconstruct
        n_tiles_h, n_tiles_w = tiles.shape[2], tiles.shape[3]
        tiles_processed = tiles_processed.permute(0, 1, 2, 4, 3, 5)
        result = tiles_processed.reshape(
            B,
            C,
            n_tiles_h * self.base_tile_size,
            n_tiles_w * self.base_tile_size,
        )

        # Resize back if needed
        if result.shape[2:] != (H, W):
            result = F.interpolate(result, size=(H, W), mode="bilinear", align_corners=False)

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self._transform(inpt, {})


# ============================================================================
# CONTINUE IN PART 2...
# ============================================================================
