# Rice Disease Classification App

This is a Flask web application that uses a TensorFlow model to classify rice diseases from uploaded images.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

## Installation

1. Clone or download this repository.
2. Navigate to the project directory.
3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Running the App

1. Ensure you are in the project directory.
2. Run the Flask application:

   ```
   python app.py
   ```

3. Open your web browser and go to `http://localhost:5000`.

## Usage

- Upload an image of a rice leaf through the web interface.
- The app will predict the disease: Bacterial leaf blight, Brown spot, or Leaf smut.
- It will display the predicted class and confidence percentage.

## Model

The model is a TensorFlow SavedModel located in the `model/1` directory. It expects input images resized to 256x256 pixels.

## Troubleshooting

- If you encounter import errors, ensure all dependencies are installed.
- Make sure the model files are present in the `model/1` directory.
- If the app doesn't start, check that port 5000 is not in use.