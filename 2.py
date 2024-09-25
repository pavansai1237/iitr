import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
import re

# Preprocessing functions
def denoise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# OCR function
def extract_text(image):
    # Convert PIL Image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocessing steps
    image = denoise(image)
    image = sharpen(image)
    gray = grayscale(image)
    thresh = threshold(gray)

    # OCR with Hindi and English language
    config = r'--oem 3 --psm 6 -l hin+eng'
    text = pytesseract.image_to_string(thresh, config=config, lang='hin+eng')

    return text

# Highlight search keywords in the text
def highlight_text(text, keyword):
    if not text or not keyword:
        return text
    # Use regex to find and wrap the keyword in bold markdown
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    highlighted_text = re.sub(pattern, lambda m: f"**{m.group(0)}**", text)
    return highlighted_text

# Create the Streamlit app
def main():
    st.title("Hindi OCR and Search with Keyword Highlight")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image and display extracted text
        extracted_text = extract_text(image)
        st.text_area("Extracted Text", extracted_text, height=200)

        # Search functionality
        keyword = st.text_input("Enter a keyword")
        if st.button("Search"):
            highlighted_text = highlight_text(extracted_text, keyword)
            st.markdown(highlighted_text)

if __name__ == "__main__":
    main()
