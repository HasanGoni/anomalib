"""Example: Adding Custom Preprocessing to Anomalib Models.

This example demonstrates how to add custom preprocessing steps (like template matching
or other torchvision transforms) to Anomalib models. The preprocessing will be:
1. Applied during training automatically
2. Embedded in the exported model
3. Used automatically during inference (no manual preprocessing needed!)

Three methods are shown:
- Method 1: Override configure_pre_processor() in a custom model
- Method 2: Pass custom PreProcessor to any existing model
- Method 3: Create a custom ONNX-compatible transform
"""

import torch
from torch import nn
from torchvision.transforms.v2 import Compose, Normalize, Resize, Transform

from anomalib.data import MVTec
from anomalib.deploy import TorchInferencer
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor


# ============================================================================
# METHOD 1: Override configure_pre_processor() in Your Model
# ============================================================================
class PadimWithCustomPreprocessing(Padim):
    """Custom Padim model with template matching preprocessing."""

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        template: torch.Tensor | None = None,
    ) -> PreProcessor:
        """Configure custom preprocessing pipeline.

        Args:
            image_size: Target image size
            template: Template for template matching (optional)

        Returns:
            PreProcessor with custom transforms
        """
        image_size = image_size or (256, 256)

        # Build custom transform pipeline
        transforms_list = [
            Resize(image_size, antialias=True),
        ]

        # Add custom preprocessing step (e.g., template matching)
        if template is not None:
            transforms_list.append(TemplateMatchingTransform(template))

        # Standard normalization (ImageNet stats)
        transforms_list.append(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

        transform = Compose(transforms_list)
        return PreProcessor(transform=transform)


# ============================================================================
# METHOD 2: Pass Custom PreProcessor to Existing Model
# ============================================================================
def use_padim_with_custom_preprocessing():
    """Use existing Padim model with custom preprocessing."""
    # Define your custom transform pipeline
    transform = Compose([
        Resize((256, 256), antialias=True),
        # Add your custom preprocessing here
        # TemplateMatchingTransform(template),
        # GaussianBlur(kernel_size=5),
        # ColorJitter(brightness=0.2, contrast=0.2),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create PreProcessor with your transforms
    pre_processor = PreProcessor(transform=transform)

    # Pass to any Anomalib model
    model = Padim(
        pre_processor=pre_processor,  # Your custom preprocessing!
        backbone="resnet18",
    )

    return model


# ============================================================================
# METHOD 3: Create Custom ONNX-Compatible Transform
# ============================================================================
class TemplateMatchingTransform(Transform):
    """Custom template matching transform (ONNX-compatible).

    This transform applies template matching as a preprocessing step.
    IMPORTANT: Use only PyTorch operations (no OpenCV) for ONNX compatibility!
    """

    def __init__(self, template: torch.Tensor) -> None:
        """Initialize template matching transform.

        Args:
            template: Template tensor for matching [C, H, W]
        """
        super().__init__()
        self.template = template

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply template matching to input.

        Args:
            inpt: Input tensor [B, C, H, W]
            params: Transform parameters (unused)

        Returns:
            Transformed tensor [B, C, H, W]
        """
        # Example: Subtract template (normalized cross-correlation could be added)
        # Use PyTorch operations only for ONNX compatibility
        if inpt.shape[1:] == self.template.shape:
            # Simple template subtraction (you can implement more sophisticated matching)
            result = torch.abs(inpt - self.template.unsqueeze(0))
        else:
            # If shapes don't match, just return input
            result = inpt

        return result

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass for the transform."""
        return self._transform(inpt, {})


class GaborFilterTransform(Transform):
    """Example: Gabor filter preprocessing (ONNX-compatible)."""

    def __init__(self, kernel_size: int = 7, sigma: float = 1.0, theta: float = 0.0) -> None:
        """Initialize Gabor filter.

        Args:
            kernel_size: Size of the Gabor kernel
            sigma: Standard deviation of Gaussian envelope
            theta: Orientation of Gabor filter
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.theta = theta

        # Pre-compute Gabor kernel (this will be part of the exported model)
        self.register_buffer("kernel", self._create_gabor_kernel())

    def _create_gabor_kernel(self) -> torch.Tensor:
        """Create Gabor kernel using PyTorch operations."""
        # Simplified Gabor kernel creation (you can make this more sophisticated)
        kernel = torch.randn(1, 1, self.kernel_size, self.kernel_size)
        return kernel

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply Gabor filter."""
        # Use conv2d for filtering (ONNX compatible)
        if inpt.dim() == 4:  # [B, C, H, W]
            # Apply to each channel
            filtered = torch.nn.functional.conv2d(
                inpt,
                self.kernel.repeat(inpt.shape[1], 1, 1, 1),
                padding=self.kernel_size // 2,
                groups=inpt.shape[1],
            )
            return filtered
        return inpt

    def forward(self, inpt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self._transform(inpt, {})


# ============================================================================
# COMPLETE TRAINING AND INFERENCE EXAMPLE
# ============================================================================
def complete_example():
    """Complete example: Train with preprocessing and use during inference."""
    print("=" * 80)
    print("CUSTOM PREPROCESSING EXAMPLE")
    print("=" * 80)

    # Step 1: Create model with custom preprocessing
    print("\n[1] Creating model with custom preprocessing...")

    # Option A: Use Method 1 (custom model)
    # template = torch.randn(3, 224, 224)  # Your template
    # model = PadimWithCustomPreprocessing(
    #     pre_processor=PadimWithCustomPreprocessing.configure_pre_processor(
    #         image_size=(256, 256),
    #         template=template,
    #     )
    # )

    # Option B: Use Method 2 (custom preprocessor)
    transform = Compose([
        Resize((256, 256), antialias=True),
        # Add your custom transforms here
        # GaborFilterTransform(kernel_size=7, sigma=1.0),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pre_processor = PreProcessor(transform=transform)
    model = Padim(pre_processor=pre_processor)

    print(f"   Model: {model.__class__.__name__}")
    print(f"   PreProcessor: {model.pre_processor}")
    print(f"   Transforms: {model.pre_processor.transform}")

    # Step 2: Setup data
    print("\n[2] Setting up data...")
    datamodule = MVTec(
        root="./datasets/MVTec",
        category="bottle",
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
    )

    # Step 3: Train (preprocessing is applied automatically!)
    print("\n[3] Training model (preprocessing applied automatically)...")
    engine = Engine(max_epochs=1)  # Use more epochs in practice
    engine.fit(model=model, datamodule=datamodule)

    print("   ✓ During training, PreProcessor applies transforms via Lightning callbacks")

    # Step 4: Export model (preprocessing is embedded!)
    print("\n[4] Exporting model (preprocessing will be embedded)...")
    export_path = "./exports/model_with_preprocessing"
    model.to_torch(export_root=export_path)

    print(f"   ✓ Model exported to: {export_path}")
    print("   ✓ Preprocessing is now part of the model!")

    # Step 5: Inference (no manual preprocessing needed!)
    print("\n[5] Running inference (no manual preprocessing needed)...")
    inferencer = TorchInferencer(
        path=f"{export_path}/model.pt",
        device="cpu",
    )

    # Just pass the raw image path - preprocessing happens automatically!
    # predictions = inferencer.predict("path/to/test_image.jpg")

    print("   ✓ Inferencer automatically applies preprocessing in model.forward()")
    print("   ✓ No need to manually preprocess images!")

    # Step 6: What happens during inference
    print("\n[6] What happens during inference:")
    print("   Raw Image → PreProcessor.forward() → Model.forward() → PostProcessor → Output")
    print("   All preprocessing is handled inside the model!")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ Preprocessing defined once in PreProcessor")
    print("✓ Applied automatically during training (via Callbacks)")
    print("✓ Embedded in exported model (via nn.Module.forward())")
    print("✓ Used automatically during inference (no manual steps!)")
    print("=" * 80)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
if __name__ == "__main__":
    # Example 1: Method 1 - Custom model
    print("\n--- Example 1: Custom Model ---")
    template = torch.randn(3, 224, 224)
    model1 = PadimWithCustomPreprocessing(
        pre_processor=PadimWithCustomPreprocessing.configure_pre_processor(
            image_size=(256, 256),
            template=template,
        ),
    )
    print(f"Model 1 preprocessor: {model1.pre_processor.transform}")

    # Example 2: Method 2 - Custom PreProcessor
    print("\n--- Example 2: Custom PreProcessor ---")
    transform = Compose([
        Resize((256, 256), antialias=True),
        GaborFilterTransform(kernel_size=7),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model2 = Padim(pre_processor=PreProcessor(transform=transform))
    print(f"Model 2 preprocessor: {model2.pre_processor.transform}")

    # Example 3: Test forward pass
    print("\n--- Example 3: Test Forward Pass ---")
    dummy_input = torch.randn(1, 3, 512, 512)  # Raw input (any size)
    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model2(dummy_input)

    print(f"Output pred_score shape: {output.pred_score.shape}")
    print(f"Output anomaly_map shape: {output.anomaly_map.shape}")
    print("\n✓ Preprocessing was applied automatically in forward()!")

    # Uncomment to run complete training/inference example
    # complete_example()
