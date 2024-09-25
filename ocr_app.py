import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load the OCR model
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return processor, model

processor, model = load_model()

def perform_ocr(image):
    try:
        # Resize the image to 384x384
        image = image.resize((384, 384))

        # Prepare the image
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process the image
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        # Generate OCR output
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text
    except Exception as e:
        st.error(f"Error during OCR: {str(e)}")
        return None

st.title("OCR for Hindi and English Text")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Perform OCR"):
        with st.spinner("Processing..."):
            extracted_text = perform_ocr(image)

        if extracted_text:
            st.success("OCR completed successfully!")
            st.subheader("Extracted Text:")
            st.text_area("Extracted Text", extracted_text, height=200)
            st.markdown(f"**Raw Extracted Text:** {extracted_text}")
        else:
            st.error("Failed to extract text from the image. Please try again with a different image.")

# Add debug information
st.sidebar.title("Debug Information")
st.sidebar.text(f"Processor loaded: {processor is not None}")
st.sidebar.text(f"Model loaded: {model is not None}")

if 'extracted_text' in locals() and extracted_text is not None:
    st.sidebar.text(f"Extracted text length: {len(extracted_text)}")
    st.sidebar.text(f"First 100 characters: {extracted_text[:100]}")
else:
    st.sidebar.text("Extracted text not available")
