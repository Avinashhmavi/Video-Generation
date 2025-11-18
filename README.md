# Video Generation

A high-performance video generation system using distilled LTX Video models from Lightricks. This repository provides optimized pipelines for fast, high-quality video generation with support for text-to-video, image-to-video, and video-to-video transformations.

## Overview

LTX Video is a state-of-the-art video generation model that uses a 3D transformer architecture with rectified flow scheduling. This implementation includes:

- **Distilled Models**: Optimized versions (2B and 13B parameters) for faster inference
- **Multi-Scale Generation**: Two-pass rendering for enhanced texture quality
- **Flexible Conditioning**: Support for image and video conditioning
- **Web Interface**: Gradio-based UI for easy interaction
- **Multiple Precision Modes**: bfloat16 and FP8 support for memory efficiency

## Features

- 🎬 **Multiple Generation Modes**
  - Text-to-video: Generate videos from text descriptions
  - Image-to-video: Animate static images
  - Video-to-video: Transform and stylize existing videos

- ⚡ **Performance Optimizations**
  - Distilled models for 2-3x faster generation
  - Multi-scale pipeline with spatial upsampling
  - Mixed precision support (bfloat16, FP8)
  - Optional CPU offloading for low-memory GPUs

- 🎨 **Quality Features**
  - Up to 257 frames generation (8.5 seconds @ 30fps)
  - Resolutions up to 1280x720
  - Spatiotemporal guidance (STG) for improved quality
  - Prompt enhancement with Florence-2 and Llama models

- 🖥️ **User-Friendly Interface**
  - Web UI powered by Gradio
  - Command-line inference script
  - Automatic dimension adjustment
  - Real-time generation progress

## Model Variants

The repository supports multiple model configurations:

| Model | Parameters | Config File | Features |
|-------|-----------|-------------|----------|
| LTXv-2B (Distilled) | 2B | `ltxv-2b-0.9.8-distilled.yaml` | Fast, efficient |
| LTXv-13B (Distilled) | 13B | `ltxv-13b-0.9.8-distilled.yaml` | Higher quality |
| LTXv-2B (FP8) | 2B | `ltxv-2b-0.9.8-distilled-fp8.yaml` | Memory optimized |
| LTXv-13B (FP8) | 13B | `ltxv-13b-0.9.8-distilled-fp8.yaml` | Memory optimized |

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU with 16GB+ VRAM (recommended)
- 32GB+ system RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Avinashhmavi/Video-Generation.git
cd ltx-video-distilled
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The requirements include:
- PyTorch with CUDA support
- Diffusers (development version)
- Transformers
- Gradio for web UI
- Image processing libraries (PIL, OpenCV, imageio)

## Usage

### Web Interface (Gradio)

Launch the interactive web interface:

```bash
python app.py
```

The interface provides:
- Three generation modes (text-to-video, image-to-video, video-to-video)
- Adjustable parameters (duration, resolution, guidance scale)
- Real-time preview and download
- Seed control for reproducibility

### Command-Line Interface

Generate videos using the inference script:

#### Text-to-Video
```bash
python inference.py \
  --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml \
  --prompt "A majestic dragon flying over a medieval castle" \
  --height 704 \
  --width 1216 \
  --num_frames 121 \
  --seed 42
```

#### Image-to-Video
```bash
python inference.py \
  --pipeline_config configs/ltxv-2b-0.9.8-distilled.yaml \
  --prompt "The creature starts to move" \
  --conditioning_media_paths input_image.jpg \
  --conditioning_start_frames 0 \
  --conditioning_strengths 1.0 \
  --height 512 \
  --width 768 \
  --num_frames 97
```

#### Video-to-Video
```bash
python inference.py \
  --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml \
  --prompt "Transform to cinematic anime style" \
  --input_media_path input_video.mp4 \
  --height 704 \
  --width 1216 \
  --num_frames 121
```

## Configuration

### Pipeline Configuration Files

Configuration files in `configs/` control model behavior:

- **checkpoint_path**: Model weights file
- **pipeline_type**: `multi-scale` or single-pass
- **downscale_factor**: First pass resolution scaling (0.66)
- **spatial_upscaler_model_path**: Upsampler model for multi-scale
- **stg_mode**: Spatiotemporal guidance strategy
  - `attention_values`: Apply guidance to attention values (default)
  - `attention_skip`: Skip layer attention
  - `residual`: Apply to residual connections
  - `transformer_block`: Block-level guidance

### Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `height` | Output video height (must be ÷32) | 512 | 256-720 |
| `width` | Output video width (must be ÷32) | 704 | 256-1280 |
| `num_frames` | Number of frames (must be N×8+1) | 121 | 9-257 |
| `frame_rate` | Frames per second | 30 | 1-60 |
| `guidance_scale` | CFG scale (prompt influence) | 3.0 | 1.0-10.0 |
| `seed` | Random seed for reproducibility | 42 | 0-2³² |

## Project Structure

```
ltx-video-distilled/
├── app.py                      # Gradio web interface
├── inference.py                # Command-line inference script
├── requirements.txt            # Python dependencies
├── configs/                    # Model configuration files
│   ├── ltxv-13b-0.9.8-distilled.yaml
│   ├── ltxv-2b-0.9.8-distilled.yaml
│   └── ...
└── ltx_video/                  # Core library
    ├── models/
    │   ├── autoencoders/      # VAE and spatial upsampler
    │   │   ├── causal_video_autoencoder.py
    │   │   ├── latent_upsampler.py
    │   │   └── vae_encode.py
    │   └── transformers/      # 3D transformer model
    │       ├── transformer3d.py
    │       ├── attention.py
    │       └── symmetric_patchifier.py
    ├── pipelines/
    │   ├── pipeline_ltx_video.py  # Main generation pipeline
    │   └── crf_compressor.py      # CRF color compression
    ├── schedulers/
    │   └── rf.py              # Rectified flow scheduler
    └── utils/
        ├── skip_layer_strategy.py  # STG implementations
        └── prompt_enhance_utils.py # Prompt enhancement
```

## Technical Details

### Architecture

**3D Transformer Model**: The core uses a spatiotemporal transformer that processes video data as 3D latent representations.

**Causal Video Autoencoder (VAE)**: Compresses video frames into latent space while maintaining temporal causality.

**Rectified Flow Scheduler**: Advanced diffusion scheduler for high-quality sampling with fewer steps.

**Latent Upsampler**: Neural upscaler that operates in latent space for the multi-scale pipeline.

### Multi-Scale Pipeline

The multi-scale generation process:

1. **First Pass**: Generate at reduced resolution (0.67×) with 7 timesteps
2. **Spatial Upsampling**: Upsample latents to target resolution
3. **Second Pass**: Refine at full resolution with 3 timesteps

This approach provides better texture detail while maintaining generation speed.

### Spatiotemporal Guidance (STG)

STG improves generation quality by applying guidance at different model layers. Four strategies are available:

- **AttentionValues**: Modifies attention mechanism outputs (recommended)
- **AttentionSkip**: Applies skip connections in attention
- **Residual**: Adds guidance to residual connections
- **TransformerBlock**: Block-level guidance application

### Memory Optimization

- **Mixed Precision**: Uses bfloat16 for reduced memory usage
- **FP8 Quantization**: Further reduces memory for supported GPUs
- **CPU Offloading**: Automatically enabled for GPUs <30GB
- **Efficient Attention**: Optimized attention mechanisms

## Advanced Features

### Prompt Enhancement

For short prompts (<120 words), automatic enhancement using:
- Florence-2 for image captioning (when conditioning on images)
- Llama-3.2-3B for cinematic prompt expansion

Disable with `enhance_prompt=False` in pipeline creation.

### Custom Conditioning

Support for multiple conditioning items with individual strengths and frame positions:

```python
conditioning_media_paths = ["frame1.jpg", "frame2.jpg"]
conditioning_start_frames = [0, 60]
conditioning_strengths = [1.0, 0.8]
```

### Video-to-Video Transformation

Transform existing videos by loading them as media items:

```python
media_tensor = load_media_file(
    media_path="input.mp4",
    height=512,
    width=768,
    max_frames=121,
    padding=(0, 0, 0, 0)
)
```

## Performance Tips

1. **Start with 2B model** for faster experimentation
2. **Use multi-scale pipeline** for final high-quality outputs
3. **Set improve_texture=False** for 2x faster preview generations
4. **Reduce num_frames** for quicker iterations (minimum: 9 frames)
5. **Keep dimensions divisible by 32** for optimal performance
6. **Use FP8 configs** if running out of memory

## Troubleshooting

### Out of Memory Errors
- Switch to smaller model variant (13B → 2B)
- Use FP8 configuration
- Reduce resolution or frame count
- Enable `offload_to_cpu=True`

### Slow Generation
- Use distilled models (2-3× faster)
- Disable multi-scale mode
- Reduce guidance scale iterations
- Use GPU with CUDA support

### Poor Quality
- Enable multi-scale pipeline
- Increase guidance scale (3.0-5.0)
- Use 13B model for better results
- Enable prompt enhancement

## Models and Weights

Models are automatically downloaded from Hugging Face Hub:
- Repository: `Lightricks/LTX-Video`
- Storage: `downloaded_models_gradio_cpu_init/` (configurable)

Manual download:
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Lightricks/LTX-Video",
    filename="ltxv-13b-0.9.8-distilled.safetensors",
    local_dir="./models"
)
```

## API Reference

### Pipeline Creation

```python
from inference import create_ltx_video_pipeline

pipeline = create_ltx_video_pipeline(
    ckpt_path="path/to/model.safetensors",
    precision="bfloat16",
    text_encoder_model_name_or_path="PixArt-alpha/PixArt-XL-2-1024-MS",
    sampler="from_checkpoint",
    device="cuda",
    enhance_prompt=False
)
```

### Generation

```python
output = pipeline(
    prompt="A cinematic video of...",
    negative_prompt="blurry, low quality",
    height=512,
    width=768,
    num_frames=97,
    frame_rate=30,
    guidance_scale=3.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
    output_type="pt"
)
```

## Acknowledgments

- **Lightricks** for the LTX Video models
- **Hugging Face** for Diffusers and model hosting
- **Gradio** for the web interface framework
- Built on PyTorch and Transformers libraries

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to Lightricks/LTX-Video for model-specific questions

---