# Quick Start: Custom Preprocessing in Anomalib

## Your Question
> "Before training, I want to use some preprocessing (e.g., template matching or torchvision transforms). How do I do that so the model container includes this preprocessing and during inference I can use it without doing any preprocessing?"

## The Answer

Anomalib has a built-in `PreProcessor` class that's **part of the model**. When you add preprocessing to a model, it's automatically:
- ✅ Applied during training
- ✅ Embedded in exported models (ONNX, OpenVINO, TorchScript)
- ✅ Used during inference (no manual preprocessing needed!)

---

## Quick Example: 3 Steps

### Step 1: Define Your Preprocessing

```python
from torchvision.transforms.v2 import Compose, Resize, Normalize
from anomalib.pre_processing import PreProcessor

# Define your custom transforms
transform = Compose([
    Resize((256, 256), antialias=True),
    # Add your custom preprocessing here!
    YourTemplateMatchingTransform(template),
    # Standard normalization
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create PreProcessor
pre_processor = PreProcessor(transform=transform)
```

### Step 2: Pass to Model

```python
from anomalib.models import Padim

# Any Anomalib model accepts pre_processor
model = Padim(pre_processor=pre_processor)
```

### Step 3: Train & Export

```python
from anomalib.engine import Engine
from anomalib.data import MVTec

# Train (preprocessing applied automatically!)
datamodule = MVTec(root="./data", category="bottle")
engine = Engine()
engine.fit(model=model, datamodule=datamodule)

# Export (preprocessing embedded!)
model.to_onnx("./exports", input_size=(256, 256))
```

### Step 4: Inference (No Manual Preprocessing!)

```python
from anomalib.deploy import ONNXInferencer

# Load model (preprocessing is inside!)
inferencer = ONNXInferencer(path="./exports/model.onnx")

# Just pass raw image - preprocessing happens automatically!
predictions = inferencer.predict("test_image.jpg")
```

---

## Architecture

```
┌─────────────────────────────────────────┐
│     AnomalibModule (Model Container)    │
├─────────────────────────────────────────┤
│  Raw Image (any size)                   │
│         ↓                                │
│  PreProcessor.forward()                 │
│    - Resize                             │
│    - Your Custom Transforms             │  ← Automatic!
│    - Normalize                          │
│         ↓                                │
│  Model.forward()                        │
│    - Feature extraction                 │
│    - Anomaly detection                  │
│         ↓                                │
│  PostProcessor.forward()                │
│    - Score normalization                │
│    - Thresholding                       │
│         ↓                                │
│  Output (scores, maps, masks)           │
└─────────────────────────────────────────┘
```

---

## Complete Examples

### Example 1: Template Matching

```python
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import Transform

class TemplateMatchingTransform(Transform):
    """ONNX-compatible template matching."""

    def __init__(self, template: torch.Tensor):
        super().__init__()
        self.register_buffer("template", template)

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        # Resize template to match input
        if inpt.shape[2:] != self.template.shape[1:]:
            template_resized = F.interpolate(
                self.template.unsqueeze(0),
                size=inpt.shape[2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            template_resized = self.template

        # Compute difference
        diff = torch.abs(inpt - template_resized.unsqueeze(0))
        return diff

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._transform(inpt, {})

# Load template
template = torch.randn(3, 256, 256)  # Replace with real template

# Create preprocessing
from torchvision.transforms.v2 import Compose, Resize, Normalize
from anomalib.pre_processing import PreProcessor

transform = Compose([
    Resize((256, 256), antialias=True),
    TemplateMatchingTransform(template),  # Your custom preprocessing!
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create model
from anomalib.models import Padim
model = Padim(pre_processor=PreProcessor(transform=transform))

# That's it! Preprocessing is now part of the model.
```

### Example 2: Edge Detection

```python
class EdgeDetectionTransform(Transform):
    """Sobel edge detection (ONNX-compatible)."""

    def __init__(self):
        super().__init__()
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ]).float().unsqueeze(0))

        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ]).float().unsqueeze(0))

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        # Convert to grayscale
        gray = inpt.mean(dim=1, keepdim=True)

        # Apply Sobel
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)

        # Compute magnitude
        edges = torch.sqrt(edge_x**2 + edge_y**2)

        # Repeat across channels
        return edges.repeat(1, inpt.shape[1], 1, 1)

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._transform(inpt, {})

# Use with model
transform = Compose([
    Resize((256, 256), antialias=True),
    EdgeDetectionTransform(),  # Your custom preprocessing!
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = Padim(pre_processor=PreProcessor(transform=transform))
```

---

## Key Rules for ONNX Compatibility

When creating custom transforms:

1. ✅ **Use PyTorch operations only** (`torch.*`, `torch.nn.functional.*`)
2. ❌ **Don't use NumPy** (`np.*`)
3. ❌ **Don't use OpenCV** (`cv2.*`)
4. ✅ **Use `register_buffer()`** for fixed tensors (templates, kernels)
5. ✅ **Avoid in-place operations** (`tensor += 1` → `tensor = tensor + 1`)
6. ✅ **Extend `torchvision.transforms.v2.Transform`**

---

## Files Created

1. **`docs/preprocessing_integration_guide.md`** - Comprehensive guide (40+ pages)
   - Detailed architecture explanation
   - Three integration methods
   - ONNX compatibility guide
   - Real-world examples
   - Testing guidelines

2. **`examples/custom_preprocessing_example.py`** - General examples
   - Method 1: Override `configure_pre_processor()`
   - Method 2: Pass custom PreProcessor
   - Method 3: Custom ONNX-compatible transforms
   - Complete training/export/inference workflow

3. **`examples/template_matching_preprocessing.py`** - Template matching specific
   - 4 template matching variants (subtraction, correlation, NCC, blending)
   - Complete working examples
   - ONNX compatibility tests
   - Utility functions for creating templates

---

## Resources

- **Detailed Guide**: `docs/preprocessing_integration_guide.md`
- **Examples**: `examples/custom_preprocessing_example.py`
- **Template Matching**: `examples/template_matching_preprocessing.py`
- **PreProcessor Source**: `src/anomalib/pre_processing/pre_processor.py`
- **CFA Model Example**: `src/anomalib/models/image/cfa/lightning_model.py` (lines 130-173)

---

## Summary

**Before:** Manual preprocessing required at every step
```python
image = preprocess(image)  # Must remember to do this!
predictions = model(image)
```

**After:** Preprocessing is part of the model
```python
model = Model(pre_processor=your_preprocessor)  # Define once
predictions = model(raw_image)  # Automatic everywhere!
```

**The preprocessing is now:**
- ✅ Automatic during training
- ✅ Embedded in exported models
- ✅ Used during inference
- ✅ Impossible to forget!

---

## Questions?

See the detailed guide in `docs/preprocessing_integration_guide.md` or the examples in `examples/`.
