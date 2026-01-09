🍎 Pomegranate Disease Detection using Deep Learning
This project leverages Convolutional Neural Networks (CNN) to identify and classify common diseases in pomegranate fruits. It features a trained deep learning model built with TensorFlow/Keras and an interactive web dashboard powered by Streamlit for real-time predictions.

🌟 Features
Automated Classification: Detects 5 distinct classes (4 diseases + Healthy).

High Accuracy: Achieved a validation accuracy of ~97.5%.

User-Friendly Interface: Upload an image via the web app and get instant results with confidence scores.

Optimized Pipeline: Includes image preprocessing (resizing to 256x256 and normalization) and efficient model caching for faster performance.

🛠️ Tech Stack
Deep Learning Framework: TensorFlow, Keras

Web App Framework: Streamlit

Data Processing: NumPy, PIL (Pillow)

Visualization: Matplotlib

Environment: Python 3.x, Jupyter Notebook

📊 Dataset & Model
Dataset Information
The model was trained on the Pomegranate Diseases Dataset, consisting of 5,000 images. The classes include:

Alternaria

Anthracnose

Bacterial Blight

Cercospora

Healthy

Model Architecture
The project utilizes a Sequential CNN architecture featuring:

Multiple Conv2D and MaxPooling2D layers for feature extraction.

A Flatten layer followed by a Dense layer (64 units, ReLU activation).

A final Softmax output layer for 5-class classification.

Total Parameters: 183,877.

🚀 Getting Started
1. Clone the Repository
Bash

git clone https://github.com/your-username/pomegranate-disease-detection.git
cd pomegranate-disease-detection
2. Install Dependencies
Create a virtual environment and install the required packages:

Bash

pip install tensorflow streamlit numpy pillow matplotlib
3. Running the App
Ensure your model (pomegranate.keras) and class names (class_names.json) are in the project root, then run:

Bash

streamlit run pomegranate_app.py
📂 Directory Structure
pomegranate disease.ipynb: The Jupyter Notebook used for data exploration, model training, and evaluation.

pomegranate_app.py: The Python script for the Streamlit web application.

pomegranate.keras: The saved trained model.

class_names.json: A JSON file mapping indices to disease labels.

📈 Performance Results
After 50 epochs of training, the model achieved:

Training Accuracy: ~98.26%

Validation Accuracy: ~97.50%

Final Loss: 0.0488

Built with ❤️ using TensorFlow & Streamlit
