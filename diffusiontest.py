from diffusers import DDPMPipeline
import torch

# Load the model first
pipeline = DDPMPipeline.from_pretrained("syzygy137/butterfly-diffusion-model")

# Then move it to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)

# Generate images
images = pipeline(
    batch_size=4,
    generator=torch.manual_seed(0)
).images

# Save the generated images
for idx, image in enumerate(images):
    image.save(f"butterfly_{idx}.png")