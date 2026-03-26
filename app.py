import joblib
import pandas as pd
from flask import Flask, request, jsonify,Flask

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load('logistic_regression_menstrual_phase_model.joblib')
label_encoder = joblib.load('label_encoder_menstrual_phase.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        # Convert input JSON to DataFrame, ensuring correct column order
        # The order of columns should match the training data (X_train)
        # Based on df.info() from earlier, the columns are:
        # age, cycle_length, day_of_cycle, avg_last_3_cycle_length, body_temp, resting_hr, hrv, n

        # You can get the column order from X.columns or X_train.columns
        # For example, if X_train.columns was ['age', 'cycle_length', ...]
        # Make sure the input JSON keys match these column names.

        # To ensure correct order and handling potential missing keys, create a DataFrame from the JSON
        # and then reindex it with the original feature columns.

        # For demonstration, assuming input will have all features in any order.
        # In a real application, you should strictly define the expected input format.

        # We need the column names from our original features (X)
        # X = df.drop('phase_label', axis=1) from earlier execution
        # We can simulate this by defining the column names manually or by accessing X.columns if X is still available.

        # Manually defining the expected feature columns based on previous `df.info()` output
        feature_columns = ['age', 'cycle_length', 'day_of_cycle', 'avg_last_3_cycle_length', 
                           'body_temp', 'resting_hr', 'hrv', 'n']

        # Create a DataFrame from the input JSON data
        input_df = pd.DataFrame([json_])

        # Reorder columns to match the training data
        input_df = input_df[feature_columns]

        # Make prediction
        prediction_encoded = model.predict(input_df)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)

        return jsonify({'prediction': prediction_label[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # In a production environment, you might use a production-ready WSGI server like Gunicorn or Waitress.
    # For Colab demonstration, we'll run it directly. Note that '0.0.0.0' makes it accessible externally if exposed.
    app.run(host='0.0.0.0', port=5000, debug=True)
