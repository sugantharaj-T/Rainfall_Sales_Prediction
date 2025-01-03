import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pre-trained model from the pickle file
with open('sales_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    temperature = float(request.form.get('temperature'))
    rainfall = float(request.form.get('rainfall'))

    # Prepare the input data for the model
    input_data = [[temperature, rainfall]]  # Model expects a 2D array

    # Make a prediction
    prediction = model.predict(input_data)[0]

    # Render the result page with the prediction
    return render_template('result.html', prediction=f"The predicted sales are: {prediction:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
