import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Load Model (Cached for Performance)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"D:\plant Disease\CNN_plant_diseases_model.keras")

model = load_model()

# Class Names (Plant Diseases)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Disease Details (Information about diseases)
disease_info = {
    'Apple___Apple_scab': {
        'description': 'Apple Scab is a fungal disease that causes dark, sunken lesions on leaves and fruit.',
        'treatment': 'Use fungicides and remove infected leaves. Prune the tree to improve airflow.'
    },
    'Apple___Black_rot': {
        'description': 'Black Rot is a fungal disease that causes black lesions on apples, resulting in soft, decayed fruit.',
        'treatment': 'Remove infected fruits and branches. Use copper-based fungicides.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Cedar Apple Rust causes yellow spots on apple leaves and can affect the quality of the fruit.',
        'treatment': 'Prune infected branches and apply fungicides regularly.'
    },
    'Apple___healthy': {
        'description': 'Healthy apple trees show no signs of disease or pest infestation.',
        'treatment': 'Ensure proper care, including regular watering, pruning, and disease prevention measures.'
    },
    'Blueberry___healthy': {
        'description': 'Healthy blueberry plants have vibrant green leaves and produce plump, healthy fruit.',
        'treatment': 'Regular maintenance and care to ensure optimal growth conditions, including appropriate watering and sunlight.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'Powdery mildew causes a white, powdery fungal growth on leaves and buds of cherry trees.',
        'treatment': 'Use fungicides, remove affected leaves, and ensure proper spacing for airflow.'
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'Healthy cherry trees are free from fungal or insect infestations, with firm, vibrant leaves.',
        'treatment': 'Regular maintenance and pruning for healthy tree growth.'
    },
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': {
        'description': 'Cercospora leaf spot results in small, round, grayish lesions on maize leaves.',
        'treatment': 'Use fungicides and practice crop rotation. Remove infected leaves.'
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Common rust causes orange pustules on the upper leaf surface of maize plants.',
        'treatment': 'Apply fungicides and remove infected plant material.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Northern leaf blight causes long, elliptical lesions on maize leaves, leading to reduced yield.',
        'treatment': 'Use resistant maize varieties, apply fungicides, and practice crop rotation.'
    },
    'Corn_(maize)___healthy': {
        'description': 'Healthy corn plants have green, vibrant leaves and strong, upright growth.',
        'treatment': 'Ensure proper watering, nutrient supply, and pest management.'
    },
    'Grape___Black_rot': {
        'description': 'Black rot causes dark, sunken lesions on grape clusters, leading to decayed fruit.',
        'treatment': 'Prune infected vines, use fungicides, and practice good vineyard management.'
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Esca is a fungal disease causing yellowing of leaves and dieback in grapevines.',
        'treatment': 'Prune affected parts and avoid over-fertilization.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Leaf blight causes irregular, dark lesions on grape leaves and can affect grape quality.',
        'treatment': 'Remove infected leaves and apply fungicides.'
    },
    'Grape___healthy': {
        'description': 'Healthy grapevines have lush green leaves and produce healthy, vibrant fruit.',
        'treatment': 'Ensure regular care, proper pruning, and disease prevention methods.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': 'Huanglongbing (HLB) causes yellowing and mottling of citrus leaves and can lead to fruit drop.',
        'treatment': 'Use systemic insecticides and remove infected trees to prevent spread.'
    },
    'Peach___Bacterial_spot': {
        'description': 'Bacterial spot causes lesions on leaves, fruit, and twigs, leading to defoliation and fruit drop.',
        'treatment': 'Use copper-based bactericides and practice proper sanitation.'
    },
    'Peach___healthy': {
        'description': 'Healthy peach trees have strong, green leaves and produce vibrant fruit.',
        'treatment': 'Maintain proper care, pruning, and disease prevention measures.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Bacterial spot causes lesions on bell pepper leaves and fruit, leading to reduced yield.',
        'treatment': 'Use bactericides and remove infected plants. Practice crop rotation.'
    },
    'Pepper,_bell___healthy': {
        'description': 'Healthy bell pepper plants have firm, green leaves and vibrant fruit.',
        'treatment': 'Ensure proper watering, pest management, and disease prevention practices.'
    },
    'Potato___Early_blight': {
        'description': 'Early blight causes dark, concentric ring lesions on potato leaves and stems.',
        'treatment': 'Use fungicides, remove infected plant material, and rotate crops.'
    },
    'Potato___Late_blight': {
        'description': 'Late blight results in large, dark lesions on potato leaves and stems, often leading to rapid decay.',
        'treatment': 'Use resistant varieties, fungicides, and crop rotation.'
    },
    'Potato___healthy': {
        'description': 'Healthy potato plants have green, unblemished leaves and strong, upright growth.',
        'treatment': 'Maintain proper care, including pest control and nutrient management.'
    },
    'Raspberry___healthy': {
        'description': 'Healthy raspberry plants show vibrant green leaves and healthy fruit growth.',
        'treatment': 'Ensure proper care, including regular pruning and disease prevention.'
    },
    'Soybean___healthy': {
        'description': 'Healthy soybean plants have green leaves and firm pods.',
        'treatment': 'Ensure proper soil, watering, and pest control practices.'
    },
    'Squash___Powdery_mildew': {
        'description': 'Powdery mildew causes white, powdery fungal growth on squash leaves and stems.',
        'treatment': 'Use fungicides and ensure proper spacing for airflow.'
    },
    'Strawberry___Leaf_scorch': {
        'description': 'Leaf scorch causes yellowing and wilting of strawberry leaves, often due to environmental stress.',
        'treatment': 'Ensure adequate irrigation and remove damaged leaves.'
    },
    'Strawberry___healthy': {
        'description': 'Healthy strawberry plants show vibrant green leaves and healthy fruit production.',
        'treatment': 'Regular care, proper watering, and pest management practices.'
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial spot causes dark lesions on tomato leaves and fruit, leading to reduced yield.',
        'treatment': 'Use copper-based bactericides and remove infected material.'
    },
    'Tomato___Early_blight': {
        'description': 'Early blight causes dark lesions with concentric rings on tomato leaves.',
        'treatment': 'Use fungicides and remove infected leaves and debris.'
    },
    'Tomato___Late_blight': {
        'description': 'Late blight results in dark, water-soaked lesions on tomato leaves and fruit.',
        'treatment': 'Use resistant varieties, fungicides, and practice crop rotation.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Leaf mold causes fuzzy mold growth on tomato leaves and can lead to defoliation.',
        'treatment': 'Use fungicides and ensure proper air circulation.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Septoria leaf spot causes round lesions with dark borders on tomato leaves.',
        'treatment': 'Use fungicides and remove affected leaves.'
    },
    'Tomato___Spider_mites_Two-spotted_spider_mite': {
        'description': 'Spider mites cause yellow speckling on tomato leaves, leading to leaf damage and drop.',
        'treatment': 'Use miticides and keep foliage dry to prevent mite spread.'
    },
    'Tomato___Target_Spot': {
        'description': 'Target spot causes circular lesions with concentric rings on tomato leaves.',
        'treatment': 'Use fungicides and remove infected leaves and debris.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Tomato Yellow Leaf Curl Virus causes yellowing and curling of tomato leaves, stunting plant growth.',
        'treatment': 'Remove infected plants and control whiteflies.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Tomato mosaic virus causes mottled, streaked leaves and reduced fruit quality.',
        'treatment': 'Remove infected plants and control aphids.'
    },
    'Tomato___healthy': {
        'description': 'Healthy tomato plants have lush green leaves and produce firm, red fruit.',
        'treatment': 'Maintain proper care, including regular watering, pruning, and pest management.'
    }
}

# Function to Process Image and Predict Disease
def model_predict(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, 224, 224, 3)

    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Sidebar Navigation
st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Choose a section", ["Home", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.image(r"C:\Users\rohit\Downloads\AI-1024x682.webp", use_column_width=True)
    st.subheader("🌱 Welcome to the Plant Disease Detection System")
    st.write("""
    This system helps farmers and agricultural professionals detect plant diseases early, ensuring healthy crops and sustainable farming practices.

    ### ✅ Benefits:
    - **Early Detection**: Identify diseases before they spread.
    - **Informed Decisions**: Take timely actions based on accurate diagnoses.
    - **Increased Yield**: Maintain the health of crops to maximize production.

    ### 🛠 Features:
    - **Accurate Predictions**: Powered by deep learning models.
    - **Easy-to-Use Interface**: Upload an image and get results instantly.
    - **Comprehensive Database**: Covers a wide range of plant diseases.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("📷 Upload an Image to Detect Plant Disease")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        # Resize the uploaded image to fit the layout
        img = Image.open(test_image)
        img = img.resize((600, 400))  # You can adjust the size (600x400) as needed
        
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict"):
            with st.spinner("🔄 Predicting..."):
                result_index = model_predict(test_image)
                prediction = class_names[result_index]
                
                # Show disease info based on prediction
                disease = disease_info.get(prediction, {})
                if 'healthy' in prediction:
                    st.write(f"🌿 Healthy: {prediction.replace('_', ' ').title()} plant.")
                    st.write(disease.get('description', 'No specific information available.'))
                else:
                    st.write(f"🌿 Unhealthy: {prediction.replace('_', ' ').title()} plant.")
                    st.write(f"**Description**: {disease.get('description', 'No information available.')}")
                    st.write(f"**Treatment**: {disease.get('treatment', 'No treatment information available.')}")