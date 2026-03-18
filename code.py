# Install required libraries
!pip install diffusers transformers accelerate torch --quiet

# Import libraries
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Text prompt
prompt = "A futuristic robot studying on a laptop in a modern classroom, cinematic lighting, ultra realistic"

# Generate image
image = pipe(prompt).images[0]

# Display image
plt.imshow(image)
plt.axis("off")

# Save image
image.save("generated_image.png")