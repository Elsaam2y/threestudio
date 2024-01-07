import argparse
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import os

"""
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python evaluate.py --model ['magic3d' or 'dreamfusion']
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["magic3d", "dreamfusion"])
args = parser.parse_args()

def evaluate_3d_model(model_name, image_folder):
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load images from the specified folder
    image_paths = [os.path.join(image_folder, image_file) for image_file in os.listdir(image_folder)]
    images = [Image.open(image_path) for image_path in image_paths]

    # Prepare inputs for CLIP model
    text = ["a DSLR photo of a medieval house"] * len(images)
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)

    # Get outputs from the CLIP model
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # image-text similarity score

    # Calculate the average score
    average_score = logits_per_image.mean().item()

    print(f"Average score for {model_name} is {average_score}")

if __name__ == "__main__":    
    if args.model == "magic3d":
        image_folder = "outputs/magic3d-coarse-if/a_zoomed_out_DSLR_photo_of_a_medieval_house@20240103-113717/evaluate_frames/"
    elif args.model == "dreamfusion":
        image_folder = "outputs/dreamfusion-if/a_zoomed_out_DSLR_photo_of_a_medieval_house@20240102-134119/evaluate_frames"
    else:
        print("Please select either 'magic3d' or 'dreamfusion' as the model argument.")
        exit()
    evaluate_3d_model(args.model, image_folder)
