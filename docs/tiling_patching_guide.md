# Tiling and Patching Preprocessing in Anomalib

This guide explains how to integrate **tiling** and **patching** preprocessing into Anomalib models so that these operations are automatically applied during both training and inference.

## Table of Contents

1. [What are Tiling and Patching?](#what-are-tiling-and-patching)
2. [Why Use Tiling/Patching Preprocessing?](#why-use-tilingpatching-preprocessing)
3. [Anomalib's Built-in Tiler](#anomalibs-built-in-tiler)
4. [Custom Tiling/Patching Transforms](#custom-tilingpatching-transforms)
5. [Integration Methods](#integration-methods)
6. [Complete Examples](#complete-examples)
7. [ONNX Compatibility](#onnx-compatibility)
8. [Comparison Table](#comparison-table)

---

## What are Tiling and Patching?

### Tiling

**Tiling** divides an image into a regular grid of non-overlapping or overlapping rectangular regions (tiles).

```
Original Image (512x512)
┌─────────────────────────┐
│  ┌─────┬─────┬─────┐   │
│  │ T1  │ T2  │ T3  │   │
│  ├─────┼─────┼─────┤   │
│  │ T4  │ T5  │ T6  │   │
│  ├─────┼─────┼─────┤   │
│  │ T7  │ T8  │ T9  │   │
│  └─────┴─────┴─────┘   │
└─────────────────────────┘
      9 tiles (each 128x128)
```

**Key Parameters:**
- **Tile size**: Size of each tile (e.g., 128x128, 256x256)
- **Stride**: Step size between tiles
  - Stride = tile_size → **non-overlapping tiles**
  - Stride < tile_size → **overlapping tiles**

### Patching

**Patching** extracts local regions from images, either:
- **Systematically**: Grid-based extraction (similar to tiling)
- **Randomly**: For data augmentation
- **Content-based**: Based on image statistics

```
Patches can be:
- Fixed-size windows (e.g., 32x32)
- Multi-scale (different sizes)
- Overlapping or non-overlapping
```

---

## Why Use Tiling/Patching Preprocessing?

| Use Case | Benefit | Example |
|----------|---------|---------|
| **Large Images** | Memory efficiency | Process 4K images by tiling into 256x256 |
| **Local Patterns** | Focus on details | Detect small defects in PCB inspection |
| **Illumination Handling** | Local normalization | Handle uneven lighting across image |
| **Data Augmentation** | Random patches | Increase training data diversity |
| **Multi-scale Analysis** | Scale invariance | Detect anomalies at different scales |
| **Computational Efficiency** | Parallel processing | Process tiles independently |

---

## Anomalib's Built-in Tiler

Anomalib includes a `Tiler` class in `src/anomalib/data/utils/tiler.py`:

### Basic Usage

```python
from anomalib.data.utils import Tiler, ImageUpscaleMode

# Create tiler
tiler = Tiler(
    tile_size=256,              # 256x256 tiles
    stride=128,                 # 128-pixel stride (overlapping)
    remove_border_count=0,      # No border removal
    mode=ImageUpscaleMode.PADDING  # Pad if image not divisible by tile_size
)

# Tile an image
import torch
image = torch.randn(1, 3, 1024, 1024)
tiles = tiler.tile(image)  # Shape: [num_tiles, 3, 256, 256]

# Process tiles...
processed_tiles = model(tiles)

# Reconstruct image
reconstructed = tiler.untile(processed_tiles)  # Shape: [1, 3, 1024, 1024]
```

### Key Features

1. **Automatic Upscaling**: Handles images not divisible by tile size
2. **Overlap Handling**: Weighted averaging in overlapping regions
3. **Random Tiling**: For data augmentation during training
4. **Border Removal**: Remove border pixels to reduce artifacts

### Using Tiler with Models

```python
from anomalib.callbacks import TilerConfigurationCallback
from anomalib.engine import Engine

# Enable tiling via callback
tiler_callback = TilerConfigurationCallback(
    enable=True,
    tile_size=512,
    stride=256,
    mode=ImageUpscaleMode.PADDING,
)

# Train with tiling
engine = Engine(callbacks=[tiler_callback])
engine.fit(model=model, datamodule=datamodule)
```

---

## Custom Tiling/Patching Transforms

For preprocessing that's **embedded in the exported model**, create custom transforms using the `PreProcessor`:

### Transform 1: Tile Extract and Process

```python
from torchvision.transforms.v2 import Transform
import torch
import torch.nn.functional as F

class TileExtractTransform(Transform):
    """Extract and process tiles (ONNX-compatible)."""

    def __init__(
        self,
        tile_size: tuple[int, int],
        stride: tuple[int, int] | None = None,
        process_mode: str = "normalize",
    ):
        super().__init__()
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size
        self.process_mode = process_mode

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        B, C, H, W = inpt.shape

        # Extract tiles using unfold (ONNX-compatible)
        tiles = inpt.unfold(2, self.tile_size[0], self.stride[0])
        tiles = tiles.unfold(3, self.tile_size[1], self.stride[1])

        # Process tiles...
        # (see full implementation in examples/tiling_patching_preprocessing.py)

        return processed_result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._transform(inpt, {})
```

### Transform 2: Patch Normalization

```python
class PatchNormalizationTransform(Transform):
    """Normalize patches independently (handles illumination variations)."""

    def __init__(self, patch_size: tuple[int, int]):
        super().__init__()
        self.patch_size = patch_size

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        # Extract patches
        patches = inpt.unfold(2, self.patch_size[0], self.patch_size[0])
        patches = patches.unfold(3, self.patch_size[1], self.patch_size[1])

        # Normalize each patch independently
        mean = patches.mean(dim=(4, 5), keepdim=True)
        std = patches.std(dim=(4, 5), keepdim=True) + 1e-8
        patches_norm = (patches - mean) / std

        # Reconstruct image
        # (reshape and interpolate back to original size)

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._transform(inpt, {})
```

### Transform 3: Multi-Scale Tiling

```python
class MultiScaleTilingTransform(Transform):
    """Process at multiple tile scales."""

    def __init__(
        self,
        tile_sizes: list[tuple[int, int]],
        aggregation: str = "mean",
    ):
        super().__init__()
        self.tile_sizes = tile_sizes
        self.aggregation = aggregation

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        results = []

        for tile_size in self.tile_sizes:
            # Extract tiles at this scale
            tiles = inpt.unfold(2, tile_size[0], tile_size[0])
            tiles = tiles.unfold(3, tile_size[1], tile_size[1])

            # Aggregate
            aggregated = tiles.mean(dim=(4, 5))

            # Upsample to original size
            upsampled = F.interpolate(
                aggregated,
                size=inpt.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            results.append(upsampled)

        # Combine scales
        return torch.stack(results).mean(dim=0)

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._transform(inpt, {})
```

---

## Integration Methods

### Method 1: Pass to PreProcessor (Recommended)

```python
from torchvision.transforms.v2 import Compose, Resize, Normalize
from anomalib.pre_processing import PreProcessor
from anomalib.models import Padim

# Create preprocessing pipeline
transform = Compose([
    Resize((256, 256), antialias=True),
    TileExtractTransform(tile_size=(64, 64), stride=(64, 64)),
    PatchNormalizationTransform(patch_size=(32, 32)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create model with tiling preprocessing
model = Padim(pre_processor=PreProcessor(transform=transform))
```

**Benefits:**
- ✅ Preprocessing embedded in exported model
- ✅ Automatic during training and inference
- ✅ Works with any Anomalib model
- ✅ ONNX-compatible when using PyTorch operations

### Method 2: Override configure_pre_processor()

```python
class PadimWithTiling(Padim):
    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        tile_size: tuple[int, int] | None = None,
    ) -> PreProcessor:
        image_size = image_size or (256, 256)
        tile_size = tile_size or (64, 64)

        transform = Compose([
            Resize(image_size, antialias=True),
            TileExtractTransform(tile_size=tile_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return PreProcessor(transform=transform)

# Use custom model
model = PadimWithTiling(
    pre_processor=PadimWithTiling.configure_pre_processor(
        image_size=(256, 256),
        tile_size=(64, 64),
    )
)
```

### Method 3: Use Built-in Tiler via Callback

```python
from anomalib.callbacks import TilerConfigurationCallback

# Enable runtime tiling
callback = TilerConfigurationCallback(
    enable=True,
    tile_size=512,
    stride=256,
)

# Train with tiling
from anomalib.engine import Engine
engine = Engine(callbacks=[callback])
engine.fit(model=model, datamodule=datamodule)
```

**Note:** This method applies tiling at the data loading stage, not as part of the model preprocessing.

---

## Complete Examples

### Example 1: Large Image Processing

**Use case:** Process 1024x1024 images on limited memory

```python
from torchvision.transforms.v2 import Compose, Resize, Normalize
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor

# Create tiling preprocessing
transform = Compose([
    Resize((1024, 1024), antialias=True),
    TileExtractTransform(
        tile_size=(256, 256),  # 256x256 tiles
        stride=(256, 256),     # Non-overlapping
        process_mode="normalize"
    ),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = Padim(pre_processor=PreProcessor(transform=transform))

# Train
engine.fit(model, datamodule)

# Export (tiling embedded!)
model.to_onnx("./exports", input_size=(1024, 1024))

# Inference (automatic tiling!)
inferencer = ONNXInferencer("./exports/model.onnx")
predictions = inferencer.predict("large_image.jpg")
```

### Example 2: Illumination Variation Handling

**Use case:** PCB inspection with uneven lighting

```python
# Create patch normalization preprocessing
transform = Compose([
    Resize((512, 512), antialias=True),
    PatchNormalizationTransform(
        patch_size=(64, 64)  # Normalize in 64x64 patches
    ),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = Padim(pre_processor=PreProcessor(transform=transform))
```

**Effect:** Each 64x64 patch normalized independently → robust to lighting variations

### Example 3: Multi-Scale Defect Detection

**Use case:** Detect anomalies at different scales

```python
# Create multi-scale preprocessing
transform = Compose([
    Resize((256, 256), antialias=True),
    MultiScaleTilingTransform(
        tile_sizes=[(32, 32), (64, 64), (128, 128)],
        aggregation="mean"
    ),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = Patchcore(pre_processor=PreProcessor(transform=transform))
```

**Effect:** Features extracted at multiple scales → scale-invariant detection

### Example 4: Combined Preprocessing Pipeline

**Use case:** Complex preprocessing with multiple steps

```python
from tiling_patching_preprocessing import (
    TileAggregationTransform,
    PatchNormalizationTransform,
    PatchContrastEnhancementTransform,
)

# Complex pipeline
transform = Compose([
    Resize((256, 256), antialias=True),

    # 1. Tile aggregation
    TileAggregationTransform(
        tile_size=(64, 64),
        stride=(32, 32),
        aggregation="mean"
    ),

    # 2. Patch normalization
    PatchNormalizationTransform(patch_size=(32, 32)),

    # 3. Contrast enhancement
    PatchContrastEnhancementTransform(
        patch_size=(32, 32),
        strength=0.5
    ),

    # 4. Standard normalization
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = Padim(pre_processor=PreProcessor(transform=transform))
```

---

## ONNX Compatibility

### Requirements for ONNX-Compatible Transforms

✅ **Use PyTorch operations only:**
- `torch.unfold()`, `torch.nn.functional.*`, `F.interpolate()`, `F.avg_pool2d()`

✅ **Avoid:**
- NumPy operations (`np.*`)
- OpenCV operations (`cv2.*`)
- Python loops with tensor-dependent conditions
- Dynamic shapes (use fixed tile sizes)

✅ **Use `register_buffer()` for fixed tensors:**
```python
class MyTransform(Transform):
    def __init__(self):
        super().__init__()
        # Register weights for ONNX export
        self.register_buffer('weights', torch.ones(64, 64))
```

### Testing ONNX Compatibility

```python
import torch
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor

# Create model with custom transforms
model = Padim(pre_processor=PreProcessor(transform=your_transform))
model.eval()

# Test forward pass
dummy_input = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    output = model(dummy_input)
print("✓ Forward pass successful")

# Test ONNX export
model.to_onnx("./test_export", input_size=(256, 256))
print("✓ ONNX export successful")

# Verify ONNX model
import onnx
onnx_model = onnx.load("./test_export/model.onnx")
onnx.checker.check_model(onnx_model)
print("✓ ONNX model is valid")
```

---

## Comparison Table

### Built-in Tiler vs Custom Transforms

| Feature | Built-in Tiler | Custom Transforms |
|---------|----------------|-------------------|
| **Integration** | Via callback | Via PreProcessor |
| **Embedded in model** | ❌ No | ✅ Yes |
| **ONNX export** | ❌ Not included | ✅ Included |
| **Flexibility** | Fixed tiling logic | Fully customizable |
| **Overlap handling** | ✅ Weighted averaging | Implement yourself |
| **When to use** | Runtime tiling | Model preprocessing |

### Tiling Strategies

| Strategy | Tile Size | Stride | Overlap | Use Case |
|----------|-----------|--------|---------|----------|
| **Non-overlapping** | 256 | 256 | 0% | Fast, no redundancy |
| **50% overlap** | 256 | 128 | 50% | Smooth boundaries |
| **75% overlap** | 256 | 64 | 75% | Very smooth, slower |
| **Sliding window** | 7 | 1 | ~85% | Local operations |

### Patch Operations

| Operation | Purpose | Example Use Case |
|-----------|---------|------------------|
| **Patch Normalization** | Handle illumination | Uneven lighting in PCB |
| **Patch Contrast Enhancement** | Enhance local details | Subtle defects |
| **Multi-Scale Patches** | Scale invariance | Varying defect sizes |
| **Adaptive Patches** | Content-based processing | Focus on high-detail regions |

---

## Common Patterns

### Pattern 1: Tile → Process → Reconstruct

```python
class TileProcessTransform(Transform):
    def _transform(self, inpt, params):
        # 1. Extract tiles
        tiles = inpt.unfold(2, tile_h, stride_h).unfold(3, tile_w, stride_w)

        # 2. Process each tile
        processed = self.process_tile(tiles)

        # 3. Reconstruct
        result = self.reconstruct(processed, inpt.shape)

        return result
```

### Pattern 2: Patch Statistics → Normalize

```python
class PatchStatisticsTransform(Transform):
    def _transform(self, inpt, params):
        # Extract patches
        patches = inpt.unfold(...)

        # Compute statistics per patch
        mean = patches.mean(dim=(4, 5), keepdim=True)
        std = patches.std(dim=(4, 5), keepdim=True)

        # Apply statistics
        patches_norm = (patches - mean) / (std + 1e-8)

        # Reconstruct
        return self.reconstruct(patches_norm)
```

### Pattern 3: Multi-Scale → Aggregate

```python
class MultiScaleTransform(Transform):
    def _transform(self, inpt, params):
        results = []

        for scale in self.scales:
            # Process at scale
            scaled_result = self.process_at_scale(inpt, scale)
            results.append(scaled_result)

        # Aggregate
        return torch.stack(results).mean(dim=0)
```

---

## Performance Considerations

### Memory Usage

| Approach | Memory | Speed | Quality |
|----------|--------|-------|---------|
| **Full image** | High | Fast | Best |
| **Non-overlapping tiles** | Low | Fast | Good |
| **50% overlapping tiles** | Medium | Medium | Better |
| **Sliding window (stride=1)** | Very High | Slow | Best |

### Recommendations

**For large images (>1024x1024):**
- Use `TileExtractTransform` with 256x256 or 512x512 tiles
- Non-overlapping or 25% overlap

**For illumination handling:**
- Use `PatchNormalizationTransform` with 32x32 or 64x64 patches

**For multi-scale defects:**
- Use `MultiScaleTilingTransform` with 3-4 scales

**For deployment:**
- Test ONNX export with target tile/patch sizes
- Verify memory usage on target device

---

## Files and Resources

### Implementation Files

- **`examples/tiling_patching_preprocessing.py`**: All transform implementations
- **`examples/tiling_patching_examples.py`**: 11 complete examples
- **`src/anomalib/data/utils/tiler.py`**: Built-in Tiler class
- **`src/anomalib/callbacks/tiler_configuration.py`**: Tiler callback

### Examples Summary

1. **Basic Tiling**: Non-overlapping tiles
2. **Overlapping Tiles**: Weighted averaging
3. **Tile Aggregation**: Statistical aggregation
4. **Patch Normalization**: Illumination handling
5. **Patch Contrast Enhancement**: Detail enhancement
6. **Sliding Window**: Local operations
7. **Multi-Scale Tiling**: Scale invariance
8. **Adaptive Tiling**: Content-based processing
9. **Complete Workflow**: Training → Export → Inference
10. **Combined Preprocessing**: Multiple techniques
11. **ONNX Compatibility**: Testing exports

---

## Quick Reference

### Basic Usage

```python
# 1. Define transform
from torchvision.transforms.v2 import Compose, Resize, Normalize
from anomalib.pre_processing import PreProcessor

transform = Compose([
    Resize((256, 256)),
    YourTilingTransform(...),  # Custom tiling/patching
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. Create model
from anomalib.models import Padim
model = Padim(pre_processor=PreProcessor(transform=transform))

# 3. Train
engine.fit(model, datamodule)

# 4. Export (tiling embedded!)
model.to_onnx("./exports", input_size=(256, 256))

# 5. Inference (automatic!)
from anomalib.deploy import ONNXInferencer
inferencer = ONNXInferencer("./exports/model.onnx")
predictions = inferencer.predict("test.jpg")  # No manual tiling!
```

---

## Summary

| Feature | Benefit |
|---------|---------|
| **Tiling/Patching as Preprocessing** | Embedded in model, automatic everywhere |
| **ONNX-Compatible Transforms** | Deploy anywhere (ONNX, OpenVINO) |
| **Multiple Strategies** | Non-overlapping, overlapping, multi-scale, adaptive |
| **Flexible Integration** | Works with any Anomalib model |
| **Memory Efficient** | Process large images in tiles |
| **Illumination Robust** | Patch-based normalization |

**The key advantage:** Define tiling/patching once → Automatic in training, export, and inference!

---

## Additional Resources

- **PreProcessor Guide**: `docs/preprocessing_integration_guide.md`
- **Template Matching**: `examples/template_matching_preprocessing.py`
- **Built-in Tiler**: `src/anomalib/data/utils/tiler.py`
- **Anomalib Docs**: https://docs.anomalib.com

---

**Need help?** Open an issue at https://github.com/openvinotoolkit/anomalib/issues
