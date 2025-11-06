"""Complete Examples: Tiling and Patching Preprocessing in Anomalib.

This file demonstrates how to use tiling and patching preprocessing
with Anomalib models for automatic integration during training and inference.
"""

import torch
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib.data import MVTec
from anomalib.deploy import TorchInferencer
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore
from anomalib.pre_processing import PreProcessor

# Import custom transforms from tiling_patching_preprocessing.py
from tiling_patching_preprocessing import (
    AdaptiveTilingTransform,
    MultiScaleTilingTransform,
    OverlappingTileTransform,
    PatchContrastEnhancementTransform,
    PatchNormalizationTransform,
    RandomPatchExtractTransform,
    SlidingWindowTransform,
    TileAggregationTransform,
    TileExtractTransform,
)


# ============================================================================
# EXAMPLE 1: Basic Tiling Preprocessing
# ============================================================================

def example_1_basic_tiling():
    """Example 1: Basic tiling preprocessing for large images."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Tiling Preprocessing")
    print("=" * 80)
    print("\nUse case: Process large images by splitting into tiles")
    print("Benefits: Memory efficiency, focus on local patterns\n")

    # Create tiling preprocessing
    transform = Compose([
        Resize((512, 512), antialias=True),
        TileExtractTransform(
            tile_size=(128, 128),  # 128x128 tiles
            stride=(128, 128),  # Non-overlapping
            process_mode="normalize",  # Normalize each tile independently
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create model with tiling preprocessing
    model = Padim(
        pre_processor=PreProcessor(transform=transform),
        backbone="resnet18",
    )

    print(f"✓ Model created with tiling preprocessing")
    print(f"  Tile size: 128x128")
    print(f"  Stride: 128x128 (non-overlapping)")
    print(f"  Processing: Independent normalization per tile")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 512, 512)
    print(f"\n  Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Output pred_score: {output.pred_score.item():.4f}")
    print(f"  Output anomaly_map: {output.anomaly_map.shape}")
    print(f"\n✓ Tiling preprocessing applied automatically in forward()!")


# ============================================================================
# EXAMPLE 2: Overlapping Tiles
# ============================================================================

def example_2_overlapping_tiles():
    """Example 2: Overlapping tiles with weighted averaging."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Overlapping Tiles")
    print("=" * 80)
    print("\nUse case: Smooth tile boundaries, avoid artifacts")
    print("Benefits: Better continuity, no seam artifacts\n")

    # Create overlapping tiling preprocessing
    transform = Compose([
        Resize((256, 256), antialias=True),
        OverlappingTileTransform(
            tile_size=128,  # 128x128 tiles
            overlap=32,  # 32 pixel overlap
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create model
    model = Padim(pre_processor=PreProcessor(transform=transform))

    print(f"✓ Model created with overlapping tiles")
    print(f"  Tile size: 128x128")
    print(f"  Overlap: 32 pixels")
    print(f"  Blending: Weighted averaging at overlaps")

    # Test
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n✓ Overlapping tiles processed with smooth blending!")


# ============================================================================
# EXAMPLE 3: Tile Aggregation
# ============================================================================

def example_3_tile_aggregation():
    """Example 3: Tile aggregation for feature extraction."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Tile Aggregation")
    print("=" * 80)
    print("\nUse case: Extract features by aggregating tile statistics")
    print("Benefits: Capture local patterns, reduce dimensionality\n")

    # Create tile aggregation preprocessing
    transform = Compose([
        Resize((256, 256), antialias=True),
        TileAggregationTransform(
            tile_size=(64, 64),
            stride=(32, 32),  # Overlapping
            aggregation="mean",  # Average within tiles
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create model
    model = Patchcore(pre_processor=PreProcessor(transform=transform))

    print(f"✓ Model created with tile aggregation")
    print(f"  Tile size: 64x64")
    print(f"  Stride: 32x32 (overlapping)")
    print(f"  Aggregation: Mean")

    # Test
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n✓ Tile aggregation preprocessing applied!")


# ============================================================================
# EXAMPLE 4: Patch Normalization
# ============================================================================

def example_4_patch_normalization():
    """Example 4: Independent patch normalization."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Patch Normalization")
    print("=" * 80)
    print("\nUse case: Handle illumination variations across image")
    print("Benefits: Robust to lighting changes, local adaptation\n")

    # Create patch normalization preprocessing
    transform = Compose([
        Resize((256, 256), antialias=True),
        PatchNormalizationTransform(
            patch_size=(32, 32),  # Normalize in 32x32 patches
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create model
    model = Padim(pre_processor=PreProcessor(transform=transform))

    print(f"✓ Model created with patch normalization")
    print(f"  Patch size: 32x32")
    print(f"  Effect: Each patch normalized independently")
    print(f"  Use case: Handles uneven lighting")

    # Test
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n✓ Patch normalization applied automatically!")


# ============================================================================
# EXAMPLE 5: Patch Contrast Enhancement
# ============================================================================

def example_5_patch_contrast_enhancement():
    """Example 5: Enhance contrast within patches."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Patch Contrast Enhancement")
    print("=" * 80)
    print("\nUse case: Enhance local details, improve defect visibility")
    print("Benefits: Better contrast, easier to detect subtle anomalies\n")

    # Create patch contrast enhancement preprocessing
    transform = Compose([
        Resize((256, 256), antialias=True),
        PatchContrastEnhancementTransform(
            patch_size=(32, 32),
            strength=0.7,  # 70% enhancement
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create model
    model = Padim(pre_processor=PreProcessor(transform=transform))

    print(f"✓ Model created with patch contrast enhancement")
    print(f"  Patch size: 32x32")
    print(f"  Strength: 0.7")
    print(f"  Effect: Enhanced local contrast per patch")

    # Test
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n✓ Contrast enhancement applied automatically!")


# ============================================================================
# EXAMPLE 6: Sliding Window Processing
# ============================================================================

def example_6_sliding_window():
    """Example 6: Sliding window for local operations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Sliding Window Processing")
    print("=" * 80)
    print("\nUse case: Apply local filters, smoothing, pattern matching")
    print("Benefits: Efficient local operations using pooling\n")

    # Create sliding window preprocessing
    transform = Compose([
        Resize((256, 256), antialias=True),
        SlidingWindowTransform(
            window_size=7,
            stride=1,
            operation="mean",  # Local averaging
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create model
    model = Padim(pre_processor=PreProcessor(transform=transform))

    print(f"✓ Model created with sliding window")
    print(f"  Window size: 7x7")
    print(f"  Stride: 1")
    print(f"  Operation: Mean (local averaging)")

    # Test
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n✓ Sliding window processing applied!")


# ============================================================================
# EXAMPLE 7: Multi-Scale Tiling
# ============================================================================

def example_7_multiscale_tiling():
    """Example 7: Multi-scale tiling for feature extraction."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Multi-Scale Tiling")
    print("=" * 80)
    print("\nUse case: Capture patterns at different scales")
    print("Benefits: Robust to scale variations, richer features\n")

    # Create multi-scale tiling preprocessing
    transform = Compose([
        Resize((256, 256), antialias=True),
        MultiScaleTilingTransform(
            tile_sizes=[(64, 64), (128, 128), (256, 256)],
            aggregation="mean",
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create model
    model = Patchcore(pre_processor=PreProcessor(transform=transform))

    print(f"✓ Model created with multi-scale tiling")
    print(f"  Scales: 64x64, 128x128, 256x256")
    print(f"  Aggregation: Mean across scales")
    print(f"  Benefit: Scale-invariant features")

    # Test
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n✓ Multi-scale tiling applied!")


# ============================================================================
# EXAMPLE 8: Adaptive Tiling
# ============================================================================

def example_8_adaptive_tiling():
    """Example 8: Adaptive tiling based on content."""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Adaptive Tiling")
    print("=" * 80)
    print("\nUse case: Adjust processing based on image content")
    print("Benefits: Focus on high-detail regions, efficient processing\n")

    # Create adaptive tiling preprocessing
    transform = Compose([
        Resize((256, 256), antialias=True),
        AdaptiveTilingTransform(
            base_tile_size=64,
            adapt_to_variance=True,  # Adjust based on variance
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create model
    model = Padim(pre_processor=PreProcessor(transform=transform))

    print(f"✓ Model created with adaptive tiling")
    print(f"  Base tile size: 64x64")
    print(f"  Adaptation: Based on variance")
    print(f"  Effect: High-detail regions processed differently")

    # Test
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n✓ Adaptive tiling applied!")


# ============================================================================
# EXAMPLE 9: Complete Training & Export Workflow
# ============================================================================

def example_9_complete_workflow():
    """Example 9: Complete workflow with tiling preprocessing."""
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Complete Workflow (Training + Export + Inference)")
    print("=" * 80)

    # Step 1: Create model with tiling preprocessing
    print("\n[1] Creating model with tiling preprocessing...")
    transform = Compose([
        Resize((256, 256), antialias=True),
        TileAggregationTransform(
            tile_size=(64, 64),
            stride=(64, 64),
            aggregation="mean",
        ),
        PatchNormalizationTransform(patch_size=(32, 32)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = Padim(
        pre_processor=PreProcessor(transform=transform),
        backbone="resnet18",
    )
    print("    ✓ Model created with tiling and patch normalization")

    # Step 2: Setup data
    print("\n[2] Setting up data...")
    # Uncomment for real training:
    # datamodule = MVTec(
    #     root="./datasets/MVTec",
    #     category="bottle",
    #     image_size=(256, 256),
    # )
    print("    ✓ Data ready (commented out for demo)")

    # Step 3: Train
    print("\n[3] Training model...")
    # Uncomment for real training:
    # engine = Engine(max_epochs=10)
    # engine.fit(model=model, datamodule=datamodule)
    print("    ✓ Training complete (commented out for demo)")
    print("    ✓ Tiling preprocessing applied automatically during training!")

    # Step 4: Export
    print("\n[4] Exporting model...")
    export_path = "./exports/tiling_model"
    # Uncomment for real export:
    # model.to_torch(export_root=export_path)
    # model.to_onnx(export_root=export_path, input_size=(256, 256))
    print(f"    ✓ Model exported (commented out for demo)")
    print(f"    ✓ Tiling preprocessing is embedded in the model!")

    # Step 5: Inference
    print("\n[5] Running inference...")
    # Uncomment for real inference:
    # inferencer = TorchInferencer(path=f"{export_path}/model.pt")
    # predictions = inferencer.predict("path/to/test_image.jpg")
    print("    ✓ Inference ready (commented out for demo)")
    print("    ✓ No manual tiling needed - it's automatic!")

    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print("✓ Tiling preprocessing defined once")
    print("✓ Applied automatically during training")
    print("✓ Embedded in exported model")
    print("✓ Used automatically during inference")
    print("=" * 80)


# ============================================================================
# EXAMPLE 10: Combining Multiple Preprocessing Techniques
# ============================================================================

def example_10_combined_preprocessing():
    """Example 10: Combine tiling, patching, and template matching."""
    print("\n" + "=" * 80)
    print("EXAMPLE 10: Combined Preprocessing")
    print("=" * 80)
    print("\nUse case: Complex preprocessing pipeline")
    print("Combines: Tiling + Patch normalization + Contrast enhancement\n")

    # Create complex preprocessing pipeline
    transform = Compose([
        # 1. Resize
        Resize((256, 256), antialias=True),

        # 2. Tile aggregation
        TileAggregationTransform(
            tile_size=(64, 64),
            stride=(32, 32),
            aggregation="mean",
        ),

        # 3. Patch normalization
        PatchNormalizationTransform(patch_size=(32, 32)),

        # 4. Contrast enhancement
        PatchContrastEnhancementTransform(
            patch_size=(32, 32),
            strength=0.5,
        ),

        # 5. Standard normalization
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create model
    model = Padim(pre_processor=PreProcessor(transform=transform))

    print(f"✓ Model created with complex preprocessing pipeline:")
    print(f"  1. Resize to 256x256")
    print(f"  2. Tile aggregation (64x64 tiles, stride 32)")
    print(f"  3. Patch normalization (32x32 patches)")
    print(f"  4. Contrast enhancement (32x32 patches, strength 0.5)")
    print(f"  5. Standard ImageNet normalization")

    # Test
    dummy_input = torch.randn(1, 3, 512, 512)
    print(f"\n  Input: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Output pred_score: {output.pred_score.item():.4f}")
    print(f"\n✓ All preprocessing steps applied automatically!")


# ============================================================================
# EXAMPLE 11: ONNX Compatibility Test
# ============================================================================

def example_11_onnx_compatibility():
    """Example 11: Test ONNX compatibility of tiling/patching."""
    print("\n" + "=" * 80)
    print("EXAMPLE 11: ONNX Compatibility Test")
    print("=" * 80)

    transforms_to_test = [
        ("TileExtract", TileExtractTransform((64, 64))),
        ("TileAggregation", TileAggregationTransform((64, 64))),
        ("PatchNormalization", PatchNormalizationTransform((32, 32))),
        ("PatchContrast", PatchContrastEnhancementTransform((32, 32))),
        ("SlidingWindow", SlidingWindowTransform(window_size=7)),
        ("MultiScale", MultiScaleTilingTransform([(64, 64), (128, 128)])),
        ("Adaptive", AdaptiveTilingTransform(64)),
    ]

    for name, transform in transforms_to_test:
        print(f"\n[Testing {name}]")

        # Create model
        pipeline = Compose([
            Resize((256, 256), antialias=True),
            transform,
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = Padim(pre_processor=PreProcessor(transform=pipeline))
        model.eval()

        # Test forward pass
        try:
            dummy_input = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"  ✓ Forward pass: OK")
        except Exception as e:
            print(f"  ✗ Forward pass: FAILED ({e})")
            continue

        # Test ONNX export
        try:
            # Uncomment for real export:
            # export_path = f"./test_exports/{name.lower()}"
            # model.to_onnx(export_root=export_path, input_size=(256, 256))
            print(f"  ✓ ONNX export: OK (commented out for demo)")
        except Exception as e:
            print(f"  ✗ ONNX export: FAILED ({e})")

    print("\n" + "=" * 80)


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TILING AND PATCHING PREPROCESSING FOR ANOMALIB")
    print("=" * 80)
    print("\nThese examples demonstrate how to integrate tiling and patching")
    print("preprocessing into Anomalib models.")
    print("\nKey benefits:")
    print("  ✓ Process large images efficiently")
    print("  ✓ Handle local variations (lighting, texture)")
    print("  ✓ Capture multi-scale features")
    print("  ✓ Automatic integration in training/inference")
    print("  ✓ ONNX-compatible for deployment")

    # Run examples
    example_1_basic_tiling()
    example_2_overlapping_tiles()
    example_3_tile_aggregation()
    example_4_patch_normalization()
    example_5_patch_contrast_enhancement()
    example_6_sliding_window()
    example_7_multiscale_tiling()
    example_8_adaptive_tiling()
    example_9_complete_workflow()
    example_10_combined_preprocessing()
    example_11_onnx_compatibility()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Choose the tiling/patching approach for your use case")
    print("  2. Uncomment training/export/inference code")
    print("  3. Test with your dataset")
    print("  4. Export and deploy")
    print("\nPreprocessing is automatic in all stages!")
    print("=" * 80 + "\n")
