import sys
import os
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Run granite-docling-258M on an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        sys.exit(1)

    try:
        # Load the model and processor
        processor = AutoProcessor.from_pretrained("ibm-granite/granite-docling-258M", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-docling-258M", trust_remote_code=True)
        
        # Open image
        image = Image.open(args.image).convert("RGB")
        
        # Prepare inputs
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate markdown
        outputs = model.generate(**inputs, max_new_tokens=1024)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        print(generated_text)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
