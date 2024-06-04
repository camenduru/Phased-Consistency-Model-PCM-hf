import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.schedulers import TCDScheduler
import spaces
from PIL import Image

SAFETY_CHECKER = True

# Constants
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
checkpoints = {
    "2-Step": ["pcm_sdxl_smallcfg_2step_converted.safetensors", 2, 0.0],
    "4-Step": ["pcm_sdxl_smallcfg_4step_converted.safetensors", 4, 0.0],
    "8-Step": ["pcm_sdxl_smallcfg_8step_converted.safetensors", 8, 0.0],
    "16-Step": ["pcm_sdxl_smallcfg_16step_converted.safetensors", 16, 0.0],
    "Normal CFG 4-Step": ["pcm_sdxl_normalcfg_4step_converted.safetensors", 4, 7.5],
    "Normal CFG 8-Step": ["pcm_sdxl_normalcfg_8step_converted.safetensors", 8, 7.5],
    "Normal CFG 16-Step": ["pcm_sdxl_normalcfg_16step_converted.safetensors", 16, 7.5],
    "LCM-Like LoRA": ["pcm_sdxl_lcmlike_lora_converted.safetensors", 16, 0.0],
}


loaded = None

# Ensure model and scheduler are initialized in GPU-enabled function
if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

if SAFETY_CHECKER:
    from safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    ).to("cuda")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    def check_nsfw_images(
        images: list[Image.Image],
    ) -> tuple[list[Image.Image], list[bool]]:
        safety_checker_input = feature_extractor(images, return_tensors="pt").to("cuda")
        has_nsfw_concepts = safety_checker(
            images=[images], clip_input=safety_checker_input.pixel_values.to("cuda")
        )

        return images, has_nsfw_concepts


# Function
@spaces.GPU(enable_queue=True)
def generate_image(prompt, ckpt):
    global loaded
    print(prompt, ckpt)

    checkpoint = checkpoints[ckpt][0]
    num_inference_steps = checkpoints[ckpt][1]
    guidance_scale = checkpoints[ckpt][2]

    if loaded != num_inference_steps:
        pipe.scheduler = TCDScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="trailing",
        )
        pipe.load_lora_weights(
            "wangfuyun/PCM_Weights", weight_name=checkpoint, subfolder="sdxl"
        )

        loaded = num_inference_steps

    results = pipe(
        prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
    )

    if SAFETY_CHECKER:
        images, has_nsfw_concepts = check_nsfw_images(results.images)
        if any(has_nsfw_concepts):
            gr.Warning("NSFW content detected.")
            return Image.new("RGB", (512, 512))
        return images[0]
    return results.images[0]


# Gradio Interface

css = """
.gradio-container {
  max-width: 60rem !important;
}
"""
with gr.Blocks(css=css) as demo:
    gr.HTML("<h1><center>SDXL-Lightning ⚡</center></h1>")
    gr.HTML(
        "<p><center>Lightning-fast text-to-image generation</center></p><p><center><a href='https://huggingface.co/ByteDance/SDXL-Lightning'>https://huggingface.co/ByteDance/SDXL-Lightning</a></center></p>"
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Textbox(label="Enter your prompt (English)", scale=8)
            ckpt = gr.Dropdown(
                label="Select inference steps",
                choices=list(checkpoints.keys()),
                value="4-Step",
                interactive=True,
            )
            submit = gr.Button(scale=1, variant="primary")
    img = gr.Image(label="SDXL-Lightning Generated Image")

    prompt.submit(
        fn=generate_image,
        inputs=[prompt, ckpt],
        outputs=img,
    )
    submit.click(
        fn=generate_image,
        inputs=[prompt, ckpt],
        outputs=img,
    )

demo.queue().launch()
