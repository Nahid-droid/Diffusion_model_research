Stable Diffusion from Scratch (PyTorch)This repository contains a purely educational, custom implementation of the Stable Diffusion inference pipeline built from scratch using PyTorch.Unlike high-level libraries (like diffusers), this project explicitly breaks down the architecture into its core components—CLIP Text Encoder, VAE (Encoder/Decoder), U-Net, and the DDPM Scheduler—making it easier to understand the inner workings of Latent Diffusion Models.📂 Project StructureThe project is modularized into the following components:Core Architectureattention.py: Implements Self-Attention and Cross-Attention mechanisms (the "transformer" parts used in CLIP and the U-Net).clip.py: The Text Encoder. Transforms text prompts into embeddings that the U-Net can understand.diffusion.py: The U-Net model with time embeddings. This predicts the noise present in the latent image at a specific timestep.encoder.py: VAE Encoder. Compresses a 512x512 pixel image into a smaller latent representation (typically 64x64).decoder.py: VAE Decoder. Reconstructs the final image from the latent representation after the diffusion process is complete.Scheduler & Pipelineddpm.py: Implements the Denoising Diffusion Probabilistic Models (DDPM) scheduler. It handles the mathematics of adding noise (forward process) and removing noise (reverse process).pipeline.py: The main orchestration script. It ties the models together to perform the generation loop, including Classifier-Free Guidance (CFG).Utilitiesmodel_loader.py: Loads the weights from a standard Stable Diffusion .ckpt file and initializes the specific sub-models.model_converter.py: Maps standard state dictionary keys to the specific variable names used in this custom implementation.Experimentsadd_noise.ipynb: Visualizes the forward diffusion process (how an image turns into noise over time).demo.ipynb: The main notebook for running Text-to-Image and Image-to-Image generation.🚀 InstallationClone the repository:git clone <repository-url>
cd <repository-folder>
Install dependencies:It is recommended to use a virtual environment.pip install -r requirements.txt
Download Model Weights:You will need a Stable Diffusion checkpoint file (e.g., v1.5 or a fine-tune like Inkpunk).Place your .ckpt file in a ../data/ folder (or adjust the path in the script).Ensure you have the tokenizer files (vocab.json and merges.txt) inside a ./data/ folder.💻 UsageBelow is a Python script adapted from demo.ipynb to run the model.import torch
from PIL import Image
from transformers import CLIPTokenizer
import model_loader
import pipeline

# 1. Setup Device
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

print(f"Using device: {DEVICE}")

# 2. Load Tokenizer and Weights
tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
model_file = "../data/Inkpunk-Diffusion-v2.ckpt" # Path to your .ckpt file
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# 3. Configure Prompt
prompt = "A futuristic cyberpunk city street at night, neon lights reflecting on wet asphalt, detailed futuristic characters, ink-style linework."
uncond_prompt = ""  # Negative prompt
do_cfg = True
cfg_scale = 8  # How strictly to follow the prompt (Min: 1, Max: 14)

# 4. Input Image (Optional - for Image-to-Image)
# Set input_image to None for standard Text-to-Image
input_image = None
# image_path = "./images/dog.jpg"
# input_image = Image.open(image_path)
strength = 0.9 # 0.0 to 1.0 (Higher = more destruction/noise added to input)

# 5. Generation parameters
sampler = "ddpm"
num_inference_steps = 50
seed = 42

# 6. Run Pipeline
output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# 7. Save/View Result
result = Image.fromarray(output_image)
result.show()
result.save("output.png")
⚙️ Parameters Explainedprompt: The text description of the image you want to generate.uncond_prompt: Also known as the "negative prompt". Things you want to avoid (e.g., "blurry, low quality").cfg_scale: Classifier-Free Guidance scale. Higher values force the model to adhere more strictly to the prompt, but too high can cause artifacts. 7-9 is usually the sweet spot.strength: Only used for Image-to-Image.Low value (e.g., 0.3): The output closely resembles the input image.High value (e.g., 0.9): The output is very different from the input image.n_inference_steps: The number of denoising steps. 50 is standard for DDPM.📋 RequirementsPython 3.11+PyTorchTransformers (Hugging Face)Pillow