# Importing essential libraries and modules

#from flask import Flask, render_template, request, Markup
from flask import Flask, render_template, request
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant-disease-model-new.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# loading fertilizer recommendation model
fertilizer_model_path = 'models/fertilizer_model.pkl'
fertilizer_model = pickle.load(open(fertilizer_model_path, 'rb'))

label_encoder_path = 'models/label_encoder.pkl'
label_encoder = pickle.load(open(label_encoder_path, 'rb'))

scaler_path = 'models/scaler.pkl'
scaler = pickle.load(open(scaler_path, 'rb'))

# Load fertilizer dataset for recommendations
fertilizer_df = pd.read_csv('Data-raw/fertilizer.csv')


# =========================================================================================

# Custom functions for calculations


import requests

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature (°C), humidity (%)
    """
    api_key = "7d180db8eaad2a2d4b7ba8eddada817f"   # <-- Tera API Key
    #base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}"
    response = requests.get(complete_url)
    data = response.json()

    if data["cod"] != "404":
        main_data = data["main"]

        temperature = round((main_data["temp"] - 273.15), 2)  # Kelvin → Celsius
        humidity = main_data["humidity"]
        return temperature, humidity
    else:
        return None



def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


def get_fertilizer_advice(N, P, K, pH, crop_name):
    """
    Get detailed fertilizer advice based on soil test results and crop requirements
    """
    try:
        # Check if crop exists in our database
        if crop_name not in label_encoder.classes_:
            similar_crops = [crop for crop in label_encoder.classes_ if crop_name.lower() in crop.lower()]
            if similar_crops:
                return {"error": f"Crop not found. Did you mean: {', '.join(similar_crops)}?"}
            return {"error": "Crop not found in database. Please check the spelling."}
        
        # Encode the crop name
        crop_encoded = label_encoder.transform([crop_name])[0]
        
        # Prepare input data
        input_data = np.array([[N, P, K, pH]])
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = fertilizer_model.predict(input_scaled)
        
        # Get the recommended values for this crop from the dataset
        crop_data = fertilizer_df[fertilizer_df['Crop'] == crop_name].iloc[0]
        
        # Calculate differences
        n_diff = crop_data['N'] - N
        p_diff = crop_data['P'] - P
        k_diff = crop_data['K'] - K
        ph_diff = crop_data['pH'] - pH
        
        # Generate advice
        advice = []
        
        # Nitrogen advice
        if n_diff > 20:
            advice.append(f"Add {n_diff} kg/ha of Nitrogen fertilizer (Urea/Ammonium Nitrate)")
        elif n_diff < -20:
            advice.append(f"Reduce Nitrogen application by {-n_diff} kg/ha")
        else:
            advice.append("Nitrogen levels are adequate")
        
        # Phosphorous advice
        if p_diff > 15:
            advice.append(f"Add {p_diff} kg/ha of Phosphorous fertilizer (DAP/SSP)")
        elif p_diff < -15:
            advice.append(f"Reduce Phosphorous application by {-p_diff} kg/ha")
        else:
            advice.append("Phosphorous levels are adequate")
        
        # Potassium advice
        if k_diff > 15:
            advice.append(f"Add {k_diff} kg/ha of Potassium fertilizer (MOP/Potassium Sulfate)")
        elif k_diff < -15:
            advice.append(f"Reduce Potassium application by {-k_diff} kg/ha")
        else:
            advice.append("Potassium levels are adequate")
        
        # pH advice
        if abs(ph_diff) > 0.5:
            if ph_diff > 0:
                advice.append(f"Soil is too acidic. Add lime to increase pH to {crop_data['pH']:.2f}")
            else:
                advice.append(f"Soil is too alkaline. Add sulfur to decrease pH to {crop_data['pH']:.2f}")
        else:
            advice.append("Soil pH is within optimal range")
        
        return {
            "crop": crop_name,
            "current_levels": {"N": N, "P": P, "K": K, "pH": pH},
            "recommended_levels": {
                "N": crop_data['N'],
                "P": crop_data['P'],
                "K": crop_data['K'],
                "pH": crop_data['pH']
            },
            "advice": advice
        }
    
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)



# render fertilizer recommendation form page
'''@app.route('/fertilizer')
    def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)'''

# render fertilizer recommendation result page
@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    pH = float(request.form['ph'])

    # Get fertilizer advice using our trained model
    result = get_fertilizer_advice(N, P, K, pH, crop_name)
    
    if "error" in result:
        return render_template('fertilizer-result.html', error=result["error"], title=title)
    
    return render_template('fertilizer-result.html', 
                         crop=result["crop"],
                         current_levels=result["current_levels"],
                         recommended_levels=result["recommended_levels"],
                         advice=result["advice"],
                         title=title)


            

    # return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
