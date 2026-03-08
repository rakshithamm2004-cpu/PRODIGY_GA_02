 PRODIGY_GA_02

## Image Generation with Pre-trained Models

This project demonstrates how to generate images from text prompts using the Stable Diffusion model.

A text description is given as input and the model generates a corresponding AI image.

## Tools Used
- Python
- Stable Diffusion
- Diffusers Library
- Google Colab
## Python code
!pip install diffusers transformers accelerate torch --quiet

from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
prompt = "A futuristic robot studying on a laptop in a modern classroom, cinematic lighting, ultra realistic"
image = pipe(prompt).images[0]

plt.imshow(image)
plt.axis("off")

image.save("generated_image.png")

## Example Prompt
"A futuristic robot studying on a laptop in a modern classroom"

## Output
The model generates an AI image based on the given text prompt.

## Internship Task
This project was completed as part of Task-02 of the Generative AI Internship at Prodigy Infotech.

## Author
 Rakshitha MM
 BSC student 
 Intern at PRODIGY INFOTECH
