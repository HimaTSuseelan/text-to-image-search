import streamlit as st
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel

def main():
    st.title("CLIP Image Search App")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    classes = ["baby shoes", "white t-shirt", "black pants", "red pants"]

    new_class = st.text_input("Enter your text to search:")
    if new_class:
        classes.append(new_class)
        index = len(classes)

        folder_path = "tiny_imaterialist/"
        max_probability = 0.0
        max_probability_image = None

        for img in os.listdir(folder_path):
            image_path = os.path.join(folder_path, img)
            image = Image.open(image_path)

            inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            if probs[0][index-1] > max_probability:
                max_probability = probs[0][index-1]
                max_probability_image = image_path

        st.write(f"Image with maximum probability: {max_probability_image} \t Probability: {max_probability}")

        image = Image.open(os.getcwd() + "/" + max_probability_image)
        st.image(image, caption=f"Image with maximum probability: {max_probability_image}", use_column_width=True)

if __name__ == "__main__":
    main()
