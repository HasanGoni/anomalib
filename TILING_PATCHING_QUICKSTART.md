# Quick Start: Tiling and Patching Preprocessing

## Your Question
> "Same thing for Tiling. Same thing for Patching."
>
> How do I integrate tiling and patching preprocessing into Anomalib models so they're automatically applied during training and inference?

## The Answer

Just like template matching, Anomalib's `PreProcessor` allows you to integrate tiling and patching as part of the model! The preprocessing is:
- ✅ Applied automatically during training
- ✅ Embedded in exported models (ONNX, OpenVINO, TorchScript)
- ✅ Used automatically during inference

---

## What are Tiling and Patching?

### Tiling
Divide images into a grid of tiles (overlapping or non-overlapping)

```
Original Image          Tiled Image
┌───────────┐          ┌─┬─┬─┐
│           │    →     ├─┼─┼─┤
│           │          └─┴─┴─┘
└───────────┘          9 tiles
```

**Use cases:**
- Process large images (4K → tiles of 256x256)
- Memory efficiency
- Parallel processing

### Patching
Extract and process local regions

```
Original Image          Patches
┌───────────┐          ┌─┐ ┌─┐ ┌─┐
│     ░     │    →     └─┘ └─┘ └─┘
│   ░░░░    │          Local regions
└───────────┘
```

**Use cases:**
- Handle illumination variations (normalize per patch)
- Focus on local patterns
- Data augmentation

---

## Quick Example: 3 Steps

### Step 1: Define Tiling/Patching Transform

```python
from torchvision.transforms.v2 import Compose, Resize, Normalize, Transform
import torch
import torch.nn.functional as F

class TileExtractTransform(Transform):
    """Extract and process tiles (ONNX-compatible)."""

    def __init__(self, tile_size: tuple[int, int], stride: tuple[int, int] | None = None):
        super().__init__()
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        # Extract tiles using unfold (ONNX-compatible)
        tiles = inpt.unfold(2, self.tile_size[0], self.stride[0])
        tiles = tiles.unfold(3, self.tile_size[1], self.stride[1])

        # Process tiles (example: normalize each tile independently)
        # ... your processing logic ...

        # Reconstruct
        # ... reconstruction logic ...

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._transform(inpt, {})
```

### Step 2: Pass to Model

```python
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor

# Create preprocessing with tiling
transform = Compose([
    Resize((256, 256), antialias=True),
    TileExtractTransform(tile_size=(64, 64), stride=(64, 64)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = Padim(pre_processor=PreProcessor(transform=transform))
```

### Step 3: Train, Export, and Infer

```python
# Train (tiling applied automatically!)
engine.fit(model, datamodule)

# Export (tiling embedded!)
model.to_onnx("./exports", input_size=(256, 256))

# Inference (automatic tiling!)
from anomalib.deploy import ONNXInferencer
inferencer = ONNXInferencer("./exports/model.onnx")
predictions = inferencer.predict("test_image.jpg")  # No manual tiling!
```

---

## Ready-to-Use Transforms

All transforms are in `examples/tiling_patching_preprocessing.py`:

### 1. TileExtractTransform
```python
TileExtractTransform(
    tile_size=(128, 128),
    stride=(128, 128),      # Non-overlapping
    process_mode="normalize"
)
```
**Use case**: Process large images in tiles

### 2. PatchNormalizationTransform
```python
PatchNormalizationTransform(
    patch_size=(32, 32)
)
```
**Use case**: Handle uneven lighting (normalize per patch)

### 3. MultiScaleTilingTransform
```python
MultiScaleTilingTransform(
    tile_sizes=[(32, 32), (64, 64), (128, 128)],
    aggregation="mean"
)
```
**Use case**: Detect anomalies at different scales

### 4. TileAggregationTransform
```python
TileAggregationTransform(
    tile_size=(64, 64),
    stride=(32, 32),        # Overlapping
    aggregation="mean"
)
```
**Use case**: Extract features by aggregating tile statistics

### 5. PatchContrastEnhancementTransform
```python
PatchContrastEnhancementTransform(
    patch_size=(32, 32),
    strength=0.7
)
```
**Use case**: Enhance local contrast for better defect visibility

### 6. OverlappingTileTransform
```python
OverlappingTileTransform(
    tile_size=128,
    overlap=32              # 32-pixel overlap
)
```
**Use case**: Smooth tile boundaries, avoid artifacts

### 7. SlidingWindowTransform
```python
SlidingWindowTransform(
    window_size=7,
    stride=1,
    operation="mean"
)
```
**Use case**: Local averaging, smoothing

### 8. AdaptiveTilingTransform
```python
AdaptiveTilingTransform(
    base_tile_size=64,
    adapt_to_variance=True
)
```
**Use case**: Adjust processing based on image content

---

## Complete Examples

### Example 1: Large Image Processing

**Problem**: Need to process 1024x1024 images but have limited memory

```python
from tiling_patching_preprocessing import TileExtractTransform

transform = Compose([
    Resize((1024, 1024), antialias=True),
    TileExtractTransform(
        tile_size=(256, 256),       # Split into 256x256 tiles
        stride=(256, 256),          # Non-overlapping
        process_mode="normalize"
    ),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = Padim(pre_processor=PreProcessor(transform=transform))
```

### Example 2: Uneven Lighting (PCB Inspection)

**Problem**: PCB images have uneven lighting across the board

```python
from tiling_patching_preprocessing import PatchNormalizationTransform

transform = Compose([
    Resize((512, 512), antialias=True),
    PatchNormalizationTransform(
        patch_size=(64, 64)         # Normalize each 64x64 patch
    ),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = Padim(pre_processor=PreProcessor(transform=transform))
```

**Effect**: Each patch normalized independently → robust to lighting variations!

### Example 3: Multi-Scale Defect Detection

**Problem**: Defects can be small or large

```python
from tiling_patching_preprocessing import MultiScaleTilingTransform

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

**Effect**: Features captured at multiple scales → detects both small and large anomalies!

### Example 4: Combined Pipeline

**Problem**: Need multiple preprocessing steps

```python
from tiling_patching_preprocessing import (
    TileAggregationTransform,
    PatchNormalizationTransform,
    PatchContrastEnhancementTransform,
)

transform = Compose([
    Resize((256, 256), antialias=True),

    # 1. Aggregate tiles
    TileAggregationTransform(
        tile_size=(64, 64),
        stride=(32, 32),
        aggregation="mean"
    ),

    # 2. Normalize patches
    PatchNormalizationTransform(patch_size=(32, 32)),

    # 3. Enhance contrast
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

## Anomalib's Built-in Tiler

Anomalib also has a built-in `Tiler` class for runtime tiling:

```python
from anomalib.callbacks import TilerConfigurationCallback

# Enable tiling via callback
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

**Note**: This applies tiling at runtime but **doesn't embed it in the exported model**.

**For embedded preprocessing**, use custom transforms with `PreProcessor` (shown above)!

---

## Built-in Tiler vs Custom Transforms

| Feature | Built-in Tiler | Custom Transforms |
|---------|----------------|-------------------|
| **Embedded in model** | ❌ No | ✅ Yes |
| **ONNX export** | ❌ Not included | ✅ Included |
| **Automatic inference** | ❌ Manual tiling needed | ✅ Automatic |
| **Flexibility** | Fixed logic | Fully customizable |
| **When to use** | Runtime tiling | Model preprocessing |

**Recommendation**: Use **custom transforms with PreProcessor** for deployment!

---

## ONNX Compatibility Rules

When creating custom tiling/patching transforms:

✅ **DO:**
- Use PyTorch operations (`torch.*`, `F.*`)
- Use `unfold()` for tile extraction
- Use `F.interpolate()` for resizing
- Use `register_buffer()` for fixed tensors

❌ **DON'T:**
- Use NumPy (`np.*`)
- Use OpenCV (`cv2.*`)
- Use Python loops with tensor conditions
- Use dynamic shapes

### Example: ONNX-Compatible Transform

```python
class MyTilingTransform(Transform):
    def __init__(self, tile_size: int):
        super().__init__()
        self.tile_size = tile_size
        # Register fixed tensors
        self.register_buffer('weights', torch.ones(tile_size, tile_size))

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        # Use PyTorch operations only
        tiles = inpt.unfold(2, self.tile_size, self.tile_size)  # ✅ PyTorch
        tiles = tiles.unfold(3, self.tile_size, self.tile_size)  # ✅ PyTorch

        # Process...
        result = F.interpolate(tiles_processed, size=inpt.shape[2:])  # ✅ PyTorch

        return result
```

---

## Comparison: Tiling vs Patching

| Aspect | Tiling | Patching |
|--------|--------|----------|
| **Structure** | Regular grid | Can be random/adaptive |
| **Coverage** | Full image | Can be partial |
| **Overlap** | Optional | Optional |
| **Use case** | Large images | Local variations |
| **Example** | 4K image → 16 tiles of 512x512 | Normalize each 32x32 patch |

**You can combine both:**
```python
transform = Compose([
    TileExtractTransform(...),       # Tiling
    PatchNormalizationTransform(...), # Patching
    Normalize(...)
])
```

---

## Files Created

1. **`examples/tiling_patching_preprocessing.py`**
   - 8+ tiling/patching transform implementations
   - All ONNX-compatible
   - Ready to use!

2. **`examples/tiling_patching_examples.py`**
   - 11 complete examples
   - Different use cases
   - Training, export, inference workflows

3. **`docs/tiling_patching_guide.md`**
   - Comprehensive guide (50+ pages)
   - Detailed explanations
   - Performance considerations
   - Comparison tables

---

## Quick Reference

```python
# 1. Import
from tiling_patching_preprocessing import TileExtractTransform
from anomalib.pre_processing import PreProcessor
from anomalib.models import Padim

# 2. Define preprocessing
transform = Compose([
    Resize((256, 256)),
    TileExtractTransform(tile_size=(64, 64)),  # Your tiling/patching!
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Create model
model = Padim(pre_processor=PreProcessor(transform=transform))

# 4. Train
engine.fit(model, datamodule)

# 5. Export (preprocessing embedded!)
model.to_onnx("./exports", input_size=(256, 256))

# 6. Inference (automatic!)
inferencer = ONNXInferencer("./exports/model.onnx")
predictions = inferencer.predict("test.jpg")  # No manual tiling!
```

---

## Summary

**Before**: Manual tiling at every step
```python
tiles = manually_tile_image(image)  # Must remember!
predictions = model(tiles)
result = manually_reconstruct(predictions)  # Must remember!
```

**After**: Tiling/Patching is part of the model
```python
model = Model(pre_processor=tiling_preprocessor)  # Define once
predictions = model(raw_image)  # Automatic everywhere!
```

**The preprocessing is now:**
- ✅ Automatic during training
- ✅ Embedded in exported models
- ✅ Used during inference
- ✅ Impossible to forget!

---

## Resources

- **Detailed Guide**: `docs/tiling_patching_guide.md`
- **Transform Implementations**: `examples/tiling_patching_preprocessing.py`
- **11 Complete Examples**: `examples/tiling_patching_examples.py`
- **Built-in Tiler**: `src/anomalib/data/utils/tiler.py`
- **Template Matching**: `examples/template_matching_preprocessing.py`
- **General Preprocessing**: `docs/preprocessing_integration_guide.md`

---

**Questions?** See the detailed guide or open an issue at https://github.com/openvinotoolkit/anomalib/issues
