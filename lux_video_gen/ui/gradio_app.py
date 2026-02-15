"""
Lux Video Generation - Gradio Web Interface.

Provides a full-featured web UI for:
- Text-to-Video generation
- Image-to-Video animation
- Video-to-Video transformation
- Multi-modal generation with audio
- Parameter tuning and preview
"""

import logging
import os
import tempfile
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def create_ui(pipeline=None, share: bool = False, server_port: int = 7860):
    """
    Create and launch the Gradio web interface.
    
    Args:
        pipeline: LuxPipeline instance (or None for demo mode)
        share: Create a public Gradio link
        server_port: Local server port
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Install gradio: pip install gradio")

    from ..inference.pipeline import GenerationConfig

    def generate_video(
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_frames: int,
        fps: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        reference_image,
        progress=gr.Progress(),
    ):
        """Generate video from UI inputs."""
        if pipeline is None:
            return None, "Pipeline not loaded. Please load a model first."

        config = GenerationConfig(
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed if seed >= 0 else None,
        )

        # Handle reference image
        ref_image_tensor = None
        if reference_image is not None:
            import numpy as np
            img = torch.from_numpy(np.array(reference_image)).float() / 255.0
            ref_image_tensor = img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        def callback(step, total, latent):
            progress(step / total, desc=f"Denoising step {step}/{total}")

        try:
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                config=config,
                reference_image=ref_image_tensor,
                callback=callback,
            )

            # Save to temporary file
            from ..inference.video_processor import VideoProcessor
            processor = VideoProcessor(target_fps=fps)

            output_path = tempfile.mktemp(suffix=".mp4")
            processor.save_video(result["video"], output_path, fps=fps)

            return output_path, f"Generated {num_frames} frames at {width}x{height}"

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None, f"Error: {str(e)}"

    # Build UI
    with gr.Blocks(
        title="Lux Video Generation",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; margin-bottom: 20px; }
        .header h1 { font-size: 2.5em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        """,
    ) as app:
        gr.HTML("""
        <div class="header">
            <h1>Lux Video Generation</h1>
            <p>Advanced AI Video Generation with Multi-Modal Support</p>
        </div>
        """)

        with gr.Tabs():
            # Text-to-Video Tab
            with gr.TabItem("Text to Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A cinematic shot of a dragon flying over mountains at sunset...",
                            lines=3,
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="blurry, low quality, artifacts...",
                            lines=2,
                        )

                        with gr.Row():
                            width = gr.Slider(256, 1920, value=512, step=64, label="Width")
                            height = gr.Slider(256, 1080, value=512, step=64, label="Height")

                        with gr.Row():
                            num_frames = gr.Slider(1, 120, value=49, step=1, label="Frames")
                            fps = gr.Slider(8, 60, value=24, step=1, label="FPS")

                        with gr.Accordion("Advanced Settings", open=False):
                            num_steps = gr.Slider(10, 100, value=50, step=5, label="Inference Steps")
                            guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                            seed = gr.Number(value=-1, label="Seed (-1 = random)")

                        generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        video_output = gr.Video(label="Generated Video")
                        status_text = gr.Textbox(label="Status", interactive=False)

                generate_btn.click(
                    fn=generate_video,
                    inputs=[prompt, negative_prompt, height, width, num_frames, fps, num_steps, guidance_scale, seed, gr.State(None)],
                    outputs=[video_output, status_text],
                )

            # Image-to-Video Tab
            with gr.TabItem("Image to Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        i2v_image = gr.Image(label="Reference Image", type="pil")
                        i2v_prompt = gr.Textbox(
                            label="Motion Prompt",
                            placeholder="The camera slowly pans right as wind blows through the trees...",
                            lines=3,
                        )
                        i2v_negative = gr.Textbox(label="Negative Prompt", lines=2)

                        with gr.Row():
                            i2v_width = gr.Slider(256, 1920, value=512, step=64, label="Width")
                            i2v_height = gr.Slider(256, 1080, value=512, step=64, label="Height")

                        with gr.Row():
                            i2v_frames = gr.Slider(1, 120, value=49, step=1, label="Frames")
                            i2v_fps = gr.Slider(8, 60, value=24, step=1, label="FPS")

                        with gr.Accordion("Advanced", open=False):
                            i2v_steps = gr.Slider(10, 100, value=50, step=5, label="Steps")
                            i2v_guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                            i2v_seed = gr.Number(value=-1, label="Seed")

                        i2v_btn = gr.Button("Animate Image", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        i2v_output = gr.Video(label="Generated Video")
                        i2v_status = gr.Textbox(label="Status", interactive=False)

                i2v_btn.click(
                    fn=generate_video,
                    inputs=[i2v_prompt, i2v_negative, i2v_height, i2v_width, i2v_frames, i2v_fps, i2v_steps, i2v_guidance, i2v_seed, i2v_image],
                    outputs=[i2v_output, i2v_status],
                )

            # Gallery / Examples Tab
            with gr.TabItem("Examples"):
                gr.Markdown("### Example Prompts")
                examples = gr.Examples(
                    examples=[
                        ["A cinematic aerial shot of a futuristic city at night with neon lights reflecting on wet streets"],
                        ["A time-lapse of flowers blooming in a garden, soft natural lighting, 4K quality"],
                        ["A slow-motion shot of ocean waves crashing against rocks, golden hour lighting"],
                        ["An astronaut floating in space with Earth visible in the background, photorealistic"],
                        ["A character walking through a enchanted forest with glowing particles and magical creatures"],
                    ],
                    inputs=[prompt] if 'prompt' in dir() else [],
                    label="Click an example to fill in the prompt",
                )

            # System Info Tab
            with gr.TabItem("System"):
                gr.Markdown("### System Information")

                def get_system_info():
                    info = []
                    info.append(f"PyTorch: {torch.__version__}")
                    info.append(f"CUDA Available: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            props = torch.cuda.get_device_properties(i)
                            mem = props.total_memory / 1e9
                            info.append(f"GPU {i}: {props.name} ({mem:.1f} GB)")
                    info.append(f"Pipeline loaded: {pipeline is not None}")
                    return "\n".join(info)

                system_info = gr.Textbox(
                    value=get_system_info,
                    label="System Info",
                    interactive=False,
                    lines=8,
                )
                gr.Button("Refresh").click(fn=get_system_info, outputs=system_info)

    # Launch
    app.queue(max_size=5)
    app.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share,
        show_error=True,
    )
    return app


if __name__ == "__main__":
    create_ui(pipeline=None, share=False)
