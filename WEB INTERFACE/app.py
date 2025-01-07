from flask import Flask, request, render_template
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image  # type: ignore
import h2o # type: ignore
from h2o.model import ModelBase # type: ignore
from h2o.frame import H2OFrame # type: ignore

# Initialize H2O
h2o.init()

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' # app.config['UPLOAD_FOLDER'] = r'C:\Users\Ayushman\Videos\MINI PROJECT\WEB INTERFACE\uploads'

# Check if the folder exists, and create it if not
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the H2O model
h2o_model_path = pickle.load(open('C:/Users/Ayushman/Videos/MINI PROJECT/WEB INTERFACE/models/NEWF_h2o_model_path.pkl', 'rb'))
data_table_model = h2o.load_model(h2o_model_path)  # Load H2O model

# Load the CNN model
cnn_model = pickle.load(open('C:/Users/Ayushman/Videos/MINI PROJECT/WEB INTERFACE/models/NEWstroke_identification_model.pkl', 'rb'))

# Function to predict from patient details
def predict_from_details(details):
    # input_data = np.array([list(details.values())])  # Convert details to NumPy array
    input_frame = H2OFrame([details])  # Convert details to H2OFrame
    column_order = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

    # Reorder the details dictionary
    details_ordered = [details[col] for col in column_order]
    
    # Convert details to H2OFrame
    input_frame = H2OFrame([details_ordered], column_names=column_order)
    
    # Convert categorical columns to factors (Ensure they are treated as categorical)
    categorical_columns = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
    for col in categorical_columns:
        input_frame[col] = input_frame[col].asfactor()
    
    prediction = data_table_model.predict(input_frame)
    return prediction['p1'].as_data_frame().iloc[0, 0] # Return the predicted class


# Function to predict from an image
def predict_from_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    prediction = cnn_model.predict(img_array)
    return prediction.argmax(axis=1)[0]  # Return the predicted class index

# Combine predictions
def combine_predictions(details, image_path=None):
    # classes = {0: 'Normal', 1: 'Stroke - Ischemic', 2: 'Stroke - Hemorrhagic'}
    classes = {0: 'Normal', 1:'Ischemic', 3:'Hemorrhagic'}
    if image_path:
        # Image prediction
        image_result = predict_from_image(image_path)
        return f"Image Prediction: {classes[image_result]}"
    else:
        # Details prediction
        details_result = predict_from_details(details)
        tt = 0.14727200772738858  # Threshold

        if details_result >= tt:
            result = "Stroke"
        else:
            result = "No Stroke"
        return f"Details Prediction: {result} (Probability: {details_result:.2%})"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():  
    # Collect patient details
    try:
        details = {
            # 'id': int(request.form['id']),
            'gender': request.form['gender'],
            'age': int(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'ever_married': request.form['ever_married'],
            'work_type': request.form['work_type'],
            'residence_type': request.form['residence_type'],
            'avg_glucose_level': float(request.form['avg_glucose_level']),
            'bmi': float(request.form['bmi']),
            'smoking_status': request.form['smoking_status'],
        }

        # Check if an image is provided
        image = request.files.get('image')
        if image and image.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            result = combine_predictions(details, image_path=image_path)
        else:
            result = combine_predictions(details)

        return f"<h1>Prediction Result : </h1><p>{result}</p>"
    except Exception as e:
        return f"<h1>Error:</h1><p>{str(e)}</p>", 400

if __name__ == '__main__':
    # Ensure the uploads folder exists
    # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True,port= 5001)

    #  pip install polars pyarrow
