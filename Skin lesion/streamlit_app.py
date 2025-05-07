
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import time
import json

# Streamlit Configuration
st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🩺",
    layout="wide"
)



import base64
import streamlit as st

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()



# Custom banner header
st.markdown("""
    <style>
        .main {
            max-width: 100% !important;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .custom-banner {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 25px;
            width: 100%;
        }
        .custom-banner h1 {
            color: #222;
            font-size: 2.5em;
            margin: 0;
        }
    </style>

    <div class="custom-banner">
        <h1>🩺 DermaScan AI: Skin Disease Classifier</h1>
    </div>
""", unsafe_allow_html=True)


# White background and font styling
st.markdown("""
    <style>
        body {
            background-color: white;
            color: black;
        }
        .reportview-container {
            background: white;
            color: black;
        }
        .css-18e3th9 {
            background-color: white;
        }
        .stMarkdown h1 {
            text-align: center;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model_accuracy.keras")

model = load_model()

# Load class labels
with open("class_labels.json") as f:
    class_labels = json.load(f)

# Disease descriptions
disease_info = {
    'akiec': "Actinic keratoses: Precancerous skin lesions caused by sun exposure",
    'bcc': "Basal cell carcinoma: Common skin cancer, rarely spreads",
    'bkl': "Benign keratosis: Non-cancerous skin growth",
    'df': "Dermatofibroma: Benign fibrous skin nodule",
    'mel': "Melanoma: Serious skin cancer requiring immediate attention",
    'nv': "Melanocytic nevi: Common benign moles",
    'vasc': "Vascular lesions: Blood vessel-related skin marks"
}

# Preprocessing for EfficientNet
def preprocess_image(img):
    img = img.resize((380, 380))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Colored bar chart
def plot_colored_bar_chart(predictions, class_labels):
    sorted_indices = np.argsort(predictions[0])[::-1]
    sorted_labels = [class_labels[i] for i in sorted_indices]
    sorted_probs = [predictions[0][i] for i in sorted_indices]
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_labels)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sorted_labels, sorted_probs, color=colors)
    ax.set_ylabel("Probability", color='black')
    ax.set_title("Predicted Class Probabilities", color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Generate image buffer for download
def generate_chart_image(predictions):
    fig = plot_colored_bar_chart(predictions, class_labels)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# Title section with styling

st.markdown("""
Upload a clear image of a skin lesion and let the AI assist with a preliminary analysis.  
**Disclaimer:** This tool is not a substitute for professional medical advice.
""")


# Upload image
uploaded_file = st.file_uploader("Upload a skin lesion image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        start_time = time.time()
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        processing_time = time.time() - start_time

    predicted_class = class_labels[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)

    with col2:
        st.subheader("Diagnosis Results")
        if confidence > 70:
            st.success(f"**{predicted_class}** (confidence: {confidence:.2f}%)")
        elif confidence > 30:
            st.warning(f"**{predicted_class}** (confidence: {confidence:.2f}%)")
        else:
            st.error(f"**{predicted_class}** (confidence: {confidence:.2f}%)")

        st.metric("Processing Time", f"{processing_time:.2f} seconds")

        st.subheader("About This Condition")
        st.info(disease_info[predicted_class])

        st.subheader("Probability Distribution")
        fig = plot_colored_bar_chart(predictions, class_labels)
        st.pyplot(fig)

        with st.expander("View Detailed Probabilities"):
            prob_df = pd.DataFrame({
                "Class": class_labels,
                "Probability": predictions[0]
            }).sort_values("Probability", ascending=False)
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

        # Download chart image
        chart_buf = generate_chart_image(predictions)
        st.download_button(
            label="Download Probability Chart (PNG)",
            data=chart_buf,
            file_name="dermascan_chart.png",
            mime="image/png"
        )
