from flask import render_template, request, jsonify
from app import app
from app.model import load_model, predict_deficiency  # Import model-related functions

# Load the ML model (this runs when the app starts)
model = load_model()

# Route to render the input form
@app.route('/input', methods=['GET', 'POST'])
def input_nutrient():
    if request.method == 'POST':
        nutrient = request.form['nutrient']
        level = float(request.form['level'])
        # Process and redirect to prediction (optional)
        return jsonify({"message": "Data received", "nutrient": nutrient, "level": level})
    return render_template('input.html')

# Route to predict deficiency based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Extract nutrient and level from the form data
    nutrient = request.form['nutrient']
    level = float(request.form['level'])
    
    # Use the model to predict deficiency
    deficiency = predict_deficiency(model, level)
    
    # Return the prediction as a JSON response
    return jsonify({"deficiency": deficiency})

