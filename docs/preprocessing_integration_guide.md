# Integrating Custom Preprocessing into Anomalib Models

This guide explains how to add custom preprocessing steps (like template matching, edge detection, Gabor filters, etc.) to Anomalib models so that preprocessing is **automatically applied during both training and inference**.

## Table of Contents

1. [How Anomalib Preprocessing Works](#how-anomalib-preprocessing-works)
2. [Why PreProcessor is Part of the Model](#why-preprocessor-is-part-of-the-model)
3. [Three Methods to Add Custom Preprocessing](#three-methods-to-add-custom-preprocessing)
4. [ONNX/OpenVINO Compatibility](#onnxopenvino-compatibility)
5. [Real-World Examples](#real-world-examples)
6. [Testing Your Preprocessing](#testing-your-preprocessing)

---

## How Anomalib Preprocessing Works

Anomalib uses a **unified preprocessing architecture** where preprocessing is part of the model container:

```
┌─────────────────────────────────────────────────────────┐
│              AnomalibModule (Model Container)            │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Raw Image Input (any size, no preprocessing)            │
│         ↓                                                 │
│  ┌─────────────────────────────┐                         │
│  │     PreProcessor            │  ← Your custom          │
│  │  - Resize                   │    transforms here!     │
│  │  - Template Matching        │                         │
│  │  - Normalize                │                         │
│  └─────────────────────────────┘                         │
│         ↓                                                 │
│  ┌─────────────────────────────┐                         │
│  │     Model (e.g., Padim)     │                         │
│  │  - Feature extraction       │                         │
│  │  - Anomaly detection        │                         │
│  └─────────────────────────────┘                         │
│         ↓                                                 │
│  ┌─────────────────────────────┐                         │
│  │     PostProcessor           │                         │
│  │  - Normalize scores         │                         │
│  │  - Apply threshold          │                         │
│  └─────────────────────────────┘                         │
│         ↓                                                 │
│  Output (pred_score, anomaly_map, pred_mask)             │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Key Insight: PreProcessor is Both a Module and Callback

The `PreProcessor` class is designed as:
- **`nn.Module`**: Used during inference and export (via `forward()` method)
- **`Lightning Callback`**: Used during training (via `on_*_batch_start()` hooks)

This dual nature ensures:
✅ Transforms are applied during training
✅ Transforms are embedded in exported models
✅ No manual preprocessing needed during inference

---

## Why PreProcessor is Part of the Model

### Traditional Approach (Manual Preprocessing)

```python
# ❌ Old way - manual preprocessing
image = load_image("test.jpg")
image = resize(image, (256, 256))
image = template_match(image, template)
image = normalize(image)
predictions = model(image)
```

**Problems:**
- Must remember to preprocess every time
- Preprocessing code separate from model
- Easy to forget steps or use wrong parameters
- Exported model doesn't include preprocessing

### Anomalib Approach (Automatic Preprocessing)

```python
# ✅ Anomalib way - automatic preprocessing
model = Padim(pre_processor=my_custom_preprocessor)

# During training: preprocessing applied automatically via callbacks
engine.fit(model, datamodule)

# During export: preprocessing embedded in model
model.to_torch("./exports")

# During inference: preprocessing happens automatically
predictions = model("test.jpg")  # No manual preprocessing!
```

**Benefits:**
- Preprocessing defined once
- Automatically applied everywhere
- Impossible to forget
- Exported model is self-contained

---

## Three Methods to Add Custom Preprocessing

### Method 1: Override `configure_pre_processor()` (Recommended for New Models)

**Best for:** Creating a new model class or standardizing preprocessing for a specific model type.

```python
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, Normalize

class PadimWithTemplateMatching(Padim):
    """Padim with custom template matching preprocessing."""

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        template: torch.Tensor | None = None,
    ) -> PreProcessor:
        """Configure preprocessing with template matching.

        Args:
            image_size: Target size for images
            template: Template for matching

        Returns:
            PreProcessor with custom transforms
        """
        image_size = image_size or (256, 256)

        transforms = [
            Resize(image_size, antialias=True),
        ]

        # Add custom preprocessing
        if template is not None:
            transforms.append(TemplateMatchingTransform(template))

        # Standard normalization
        transforms.append(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

        return PreProcessor(transform=Compose(transforms))

# Usage
template = load_template("template.jpg")
model = PadimWithTemplateMatching(
    pre_processor=PadimWithTemplateMatching.configure_pre_processor(
        image_size=(256, 256),
        template=template,
    )
)
```

**Advantages:**
- Clean API for model-specific preprocessing
- Easy to document and share
- Follows Anomalib conventions (see CFA model)

**Example:** See `src/anomalib/models/image/cfa/lightning_model.py:130-173`

---

### Method 2: Pass Custom PreProcessor to Constructor (Recommended for Quick Experiments)

**Best for:** Trying different preprocessing pipelines with existing models.

```python
from anomalib.models import Padim, Patchcore, EfficientAd
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, Normalize, GaussianBlur

# Define custom preprocessing pipeline
transform = Compose([
    Resize((256, 256), antialias=True),
    GaussianBlur(kernel_size=5, sigma=1.0),
    # Add your custom transforms here
    TemplateMatchingTransform(template),
    EdgeDetectionTransform(),
    # Standard normalization
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create preprocessor
pre_processor = PreProcessor(transform=transform)

# Use with ANY Anomalib model
model = Padim(pre_processor=pre_processor)
# or
model = Patchcore(pre_processor=pre_processor)
# or
model = EfficientAd(pre_processor=pre_processor)
```

**Advantages:**
- Flexible and quick
- Works with any Anomalib model
- Easy to experiment with different transforms
- No need to create new model classes

---

### Method 3: Create Custom ONNX-Compatible Transforms (Required for Complex Preprocessing)

**Best for:** Implementing custom preprocessing logic that needs to work with ONNX/OpenVINO export.

```python
from torchvision.transforms.v2 import Transform
import torch
import torch.nn.functional as F

class TemplateMatchingTransform(Transform):
    """Template matching preprocessing (ONNX-compatible).

    IMPORTANT: Use only PyTorch operations (no OpenCV, NumPy, etc.)
    for ONNX compatibility!
    """

    def __init__(self, template: torch.Tensor, threshold: float = 0.8):
        """Initialize template matching.

        Args:
            template: Template tensor [C, H, W]
            threshold: Matching threshold
        """
        super().__init__()
        self.template = template
        self.threshold = threshold

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply template matching.

        Args:
            inpt: Input tensor [B, C, H, W]
            params: Transform parameters (unused)

        Returns:
            Transformed tensor [B, C, H, W]
        """
        # Use PyTorch operations only
        # Example: Normalized cross-correlation using conv2d

        # Normalize input
        inpt_norm = F.normalize(inpt, p=2, dim=1)
        template_norm = F.normalize(self.template.unsqueeze(0), p=2, dim=1)

        # Compute correlation (simplified - you can make this more sophisticated)
        result = inpt_norm * template_norm

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass for the transform."""
        return self._transform(inpt, {})


class EdgeDetectionTransform(Transform):
    """Sobel edge detection (ONNX-compatible)."""

    def __init__(self):
        super().__init__()
        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ]).float().unsqueeze(0))

        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        ]).float().unsqueeze(0))

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply edge detection."""
        # Convert to grayscale (average across channels)
        gray = inpt.mean(dim=1, keepdim=True)

        # Apply Sobel filters
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)

        # Compute magnitude
        edges = torch.sqrt(edge_x**2 + edge_y**2)

        # Repeat across channels
        edges = edges.repeat(1, inpt.shape[1], 1, 1)

        return edges

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._transform(inpt, {})
```

**Critical Requirements for ONNX Compatibility:**
1. ✅ Use only PyTorch operations (`torch.`, `torch.nn.functional`)
2. ✅ Avoid NumPy operations
3. ✅ Avoid OpenCV operations
4. ✅ Avoid Python control flow (if/for with tensor conditions)
5. ✅ Use `register_buffer()` for fixed tensors (kernels, templates)
6. ✅ Test export with `model.to_onnx()` before deployment

---

## ONNX/OpenVINO Compatibility

### Automatic Transform Conversion

Anomalib automatically converts transforms for export compatibility:

**File:** `src/anomalib/pre_processing/utils/transform.py`

```python
def get_exportable_transform(transform):
    """Convert transforms for ONNX/OpenVINO export.

    - Disables antialiasing in Resize (not ONNX compatible)
    - Converts CenterCrop to ExportableCenterCrop
    - Deep copies transform to avoid modifying original
    """
    transform = copy.deepcopy(transform)
    transform = disable_antialiasing(transform)
    transform = convert_center_crop_transform(transform)
    return transform
```

### Export Process

When you export a model:

```python
# Export to PyTorch
model.to_torch("./exports")
# Saves: AnomalibModule with PreProcessor, Model, PostProcessor

# Export to ONNX
model.to_onnx("./exports", input_size=(256, 256))
# Traces: PreProcessor.forward() → Model.forward() → PostProcessor.forward()

# Export to OpenVINO
model.to_openvino("./exports", input_size=(256, 256))
# Converts ONNX → OpenVINO IR format
```

The exported model includes the **complete pipeline**:
```
Raw Image → PreProcessor → Model → PostProcessor → Output
```

### Testing ONNX Compatibility

```python
import torch
import onnx
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor

# Create model with custom preprocessing
model = Padim(pre_processor=your_preprocessor)
model.eval()

# Test forward pass
dummy_input = torch.randn(1, 3, 256, 256)
output = model(dummy_input)
print(f"✓ Forward pass successful: {output.pred_score.shape}")

# Test ONNX export
export_path = "./test_export"
model.to_onnx(export_path, input_size=(256, 256))
print(f"✓ ONNX export successful: {export_path}")

# Verify ONNX model
onnx_model = onnx.load(f"{export_path}/model.onnx")
onnx.checker.check_model(onnx_model)
print("✓ ONNX model is valid")

# Test inference with ONNX
from anomalib.deploy import ONNXInferencer
inferencer = ONNXInferencer(path=f"{export_path}/model.onnx")
predictions = inferencer.predict("test_image.jpg")
print(f"✓ ONNX inference successful: {predictions.pred_score}")
```

---

## Real-World Examples

### Example 1: Template Matching for PCB Inspection

```python
import torch
from torchvision.transforms.v2 import Transform, Compose, Resize, Normalize
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor

class PCBTemplateMatchingTransform(Transform):
    """Template matching for PCB anomaly detection."""

    def __init__(self, template: torch.Tensor, mode: str = "subtract"):
        super().__init__()
        self.template = template
        self.mode = mode

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply template matching."""
        # Ensure template matches input shape
        if inpt.shape[2:] != self.template.shape[1:]:
            template_resized = F.interpolate(
                self.template.unsqueeze(0),
                size=inpt.shape[2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            template_resized = self.template

        if self.mode == "subtract":
            # Highlight differences from template
            result = torch.abs(inpt - template_resized.unsqueeze(0))
        elif self.mode == "correlation":
            # Normalized correlation
            result = F.cosine_similarity(
                inpt.view(inpt.shape[0], -1),
                template_resized.view(1, -1),
                dim=1
            ).view(inpt.shape[0], 1, 1, 1).expand_as(inpt)

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._transform(inpt, {})

# Load reference template
template = torch.load("reference_pcb_template.pt")  # [C, H, W]

# Create preprocessing pipeline
transform = Compose([
    Resize((256, 256), antialias=True),
    PCBTemplateMatchingTransform(template, mode="subtract"),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create model
model = Padim(
    pre_processor=PreProcessor(transform=transform),
    backbone="resnet18",
)

# Train and export
from anomalib.engine import Engine
from anomalib.data import MVTec

datamodule = MVTec(root="./data", category="transistor")
engine = Engine()
engine.fit(model, datamodule)

# Export with preprocessing embedded
model.to_onnx("./pcb_model", input_size=(256, 256))
```

### Example 2: Edge-Enhanced Anomaly Detection

```python
class EdgeEnhancementTransform(Transform):
    """Enhance edges for texture anomaly detection."""

    def __init__(self, edge_weight: float = 0.3):
        super().__init__()
        self.edge_weight = edge_weight

        # Laplacian kernel for edge detection
        self.register_buffer('laplacian', torch.tensor([
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]]
        ]).float().unsqueeze(0))

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply edge enhancement."""
        # Average across channels for edge detection
        gray = inpt.mean(dim=1, keepdim=True)

        # Detect edges
        edges = F.conv2d(gray, self.laplacian, padding=1)
        edges = torch.abs(edges)

        # Repeat across channels
        edges = edges.repeat(1, inpt.shape[1], 1, 1)

        # Blend original with edges
        result = inpt + self.edge_weight * edges

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._transform(inpt, {})

# Use for texture anomaly detection
transform = Compose([
    Resize((256, 256), antialias=True),
    EdgeEnhancementTransform(edge_weight=0.3),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = Patchcore(pre_processor=PreProcessor(transform=transform))
```

### Example 3: Multi-Scale Preprocessing

```python
class MultiScaleTransform(Transform):
    """Multi-scale preprocessing for capturing details at different resolutions."""

    def __init__(self, scales: list[float] = [0.5, 1.0, 2.0]):
        super().__init__()
        self.scales = scales

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Create multi-scale representation."""
        B, C, H, W = inpt.shape

        # Compute at multiple scales
        multi_scale_features = []
        for scale in self.scales:
            if scale != 1.0:
                scaled = F.interpolate(
                    inpt,
                    scale_factor=scale,
                    mode='bilinear',
                    align_corners=False
                )
                # Resize back to original
                scaled = F.interpolate(
                    scaled,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled = inpt
            multi_scale_features.append(scaled)

        # Concatenate or average
        result = torch.stack(multi_scale_features).mean(dim=0)

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._transform(inpt, {})
```

---

## Testing Your Preprocessing

### Unit Test Template

```python
import torch
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor

def test_custom_preprocessing():
    """Test custom preprocessing integration."""

    # 1. Create model with custom preprocessing
    transform = Compose([
        Resize((256, 256), antialias=True),
        YourCustomTransform(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = Padim(pre_processor=PreProcessor(transform=transform))
    model.eval()

    # 2. Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)  # Batch of 2, any size
    with torch.no_grad():
        output = model(dummy_input)

    assert output.pred_score.shape == (2,), "Incorrect output shape"
    assert output.anomaly_map.shape[0] == 2, "Incorrect anomaly map batch size"
    print("✓ Forward pass test passed")

    # 3. Test ONNX export
    export_path = "./test_export"
    model.to_onnx(export_path, input_size=(256, 256))
    assert (Path(export_path) / "model.onnx").exists(), "ONNX export failed"
    print("✓ ONNX export test passed")

    # 4. Test ONNX inference
    from anomalib.deploy import ONNXInferencer
    inferencer = ONNXInferencer(path=f"{export_path}/model.onnx")

    # Create dummy image file
    test_image = torch.randn(3, 512, 512)
    torchvision.utils.save_image(test_image, "test_image.jpg")

    predictions = inferencer.predict("test_image.jpg")
    assert predictions.pred_score is not None, "Inference failed"
    print("✓ ONNX inference test passed")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_custom_preprocessing()
```

---

## Common Pitfalls and Solutions

### ❌ Pitfall 1: Using Non-ONNX Operations

```python
# ❌ BAD - Uses OpenCV (not ONNX compatible)
import cv2

class BadTransform(Transform):
    def _transform(self, inpt, params):
        inpt_np = inpt.cpu().numpy()
        result = cv2.GaussianBlur(inpt_np, (5, 5), 0)  # ❌ OpenCV!
        return torch.from_numpy(result)
```

```python
# ✅ GOOD - Uses PyTorch operations
class GoodTransform(Transform):
    def _transform(self, inpt, params):
        result = F.gaussian_blur(inpt, kernel_size=[5, 5])  # ✅ PyTorch!
        return result
```

### ❌ Pitfall 2: Forgetting to Export Preprocessing

```python
# ❌ BAD - Preprocessing not part of model
def preprocess(image):
    return resize_and_normalize(image)

image = preprocess(load_image("test.jpg"))  # Manual preprocessing
model(image)  # Model doesn't know about preprocessing
```

```python
# ✅ GOOD - Preprocessing part of model
model = Padim(pre_processor=PreProcessor(transform=my_transforms))
model(load_image("test.jpg"))  # Preprocessing happens automatically
```

### ❌ Pitfall 3: Modifying Tensors In-Place

```python
# ❌ BAD - In-place operations can break ONNX tracing
class BadTransform(Transform):
    def _transform(self, inpt, params):
        inpt += 1.0  # ❌ In-place modification
        return inpt
```

```python
# ✅ GOOD - Create new tensor
class GoodTransform(Transform):
    def _transform(self, inpt, params):
        result = inpt + 1.0  # ✅ New tensor
        return result
```

---

## Summary

| Feature | Benefit |
|---------|---------|
| **PreProcessor as nn.Module** | Preprocessing embedded in exported models |
| **PreProcessor as Callback** | Automatic preprocessing during training |
| **ONNX-compatible transforms** | Deploy anywhere (ONNX, OpenVINO, TorchScript) |
| **Three integration methods** | Flexibility for different use cases |
| **Unified architecture** | Preprocessing defined once, used everywhere |

### Quick Reference

```python
# Define transforms
transform = Compose([
    Resize((256, 256)),
    YourCustomTransform(),  # Template matching, edge detection, etc.
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create model
model = AnyAnomalibModel(pre_processor=PreProcessor(transform=transform))

# Train (preprocessing automatic)
engine.fit(model, datamodule)

# Export (preprocessing embedded)
model.to_onnx("./exports", input_size=(256, 256))

# Inference (preprocessing automatic)
inferencer = ONNXInferencer("./exports/model.onnx")
predictions = inferencer.predict("test.jpg")  # No manual preprocessing!
```

---

## Additional Resources

- **PreProcessor implementation**: `src/anomalib/pre_processing/pre_processor.py`
- **Transform utilities**: `src/anomalib/pre_processing/utils/transform.py`
- **Example (CFA model)**: `src/anomalib/models/image/cfa/lightning_model.py:130-173`
- **Export functionality**: `src/anomalib/models/components/base/export_mixin.py`
- **Example code**: `examples/custom_preprocessing_example.py`

---

**Need help?** Open an issue at https://github.com/openvinotoolkit/anomalib/issues
