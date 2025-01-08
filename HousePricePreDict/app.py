from flask import Flask, render_template, request
import numpy as np
import pickle
from keras.models import load_model

app = Flask(__name__)

# Load the model and scalers
model = load_model('house_price_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
streets = pickle.load(open('my_list.pkl', 'rb'))
y_scaler = pickle.load(open('y_scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        floors = int(request.form['floors'])
        street_input = request.form['street']

        # One-hot encode the street input
        street_one_hot = [1 if street_input == street else 0 for street in streets]

        # Create the feature vector
        new_data = np.array([[area, bedrooms, bathrooms, floors] + street_one_hot])

        # Scale the input data
        new_data_scaled = scaler.transform(new_data)

        # Make a prediction
        predicted_price = model.predict(new_data_scaled)[0][0]

        # Reverse the scaling for the predicted price
        predicted_price_unscaled = y_scaler.inverse_transform([[predicted_price]])[0][0]

        # Return the predicted price to the user
        return render_template('index.html', predicted_price=predicted_price_unscaled)

    return render_template('index.html', predicted_price=None)

if __name__ == '__main__':
    app.run(debug=True)
