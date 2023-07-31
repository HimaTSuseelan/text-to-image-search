import streamlit as st
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel

def main():
    st.set_page_config(layout="wide")  # Set Streamlit to use full page width
    st.title("Image Search App")

    # Create a two-column layout
    col1, col2 = st.columns([2, 3])

    folder_path = "tiny_imaterialist/"
    images = []  # List to store all the images in the folder

    for img in os.listdir(folder_path):
        image_path = os.path.join(folder_path, img)
        image = Image.open(image_path)
        image = image.resize((100, 100))
        images.append(image)

    # Display all the images in the folder on the left side (col1)
    st.write("Images present in database")

    col1.image(images, caption=[f"Image {i+1}" for i in range(len(images))], width=100)

    # In the right column (col2), display the image search input and results
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    classes = ["baby shoes", "white t-shirt", "black pants", "red pants"]

    new_class = col2.text_input("Enter your text to search:")
    if new_class:
        classes.append(new_class)
        index = len(classes)

        top_images = []  # List to store the top four images with the highest probabilities
        num_images_to_display = 4

        for img in os.listdir(folder_path):
            image_path = os.path.join(folder_path, img)
            image = Image.open(image_path)

            inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            probability = probs[0][index-1].item()

            # Keep only top num_images_to_display images with the highest probabilities
            if len(top_images) < num_images_to_display:
                top_images.append((probability, image_path))
            else:
                min_probability_index = min(range(len(top_images)), key=lambda i: top_images[i][0])
                if probability > top_images[min_probability_index][0]:
                    top_images[min_probability_index] = (probability, image_path)

        top_images = sorted(top_images, reverse=True)

        # Display the top four images with the highest probabilities on the right side (col2)
        with col2:
            st.write("Top", num_images_to_display, "Images with Highest Probabilities:")
            for prob, image_path in top_images:
                st.write(f"Probability: {prob}")

                image = Image.open(image_path)
                if(image.width > image.height):
                    new_width = 300
                    new_height = 200
                else:
                    new_width = 200
                    new_height = 300
                image = image.resize((new_width, new_height))

                st.image(image, caption=f"Image with probability: {prob}")

if __name__ == "__main__":
    main()
