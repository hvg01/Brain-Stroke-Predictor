from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle


app = Flask(__name__, template_folder='templates')

app = Flask(__name__)
model = pickle.load(open('stroke_changed.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
	int_features = [float(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(scaler.transform(final_features))

	if prediction == 0:
		output = 'Prediction: You do not have a chance of suffering from a stroke. But, for a safer side you may consult a doctor within 15-20 days.'
	if prediction == 1:
		output = 'Prediction: You have a chance of suffering from a stroke. Please, consult a doctor immediately.'

	return render_template('main.html', prediction_text = output,prediction=prediction)


if __name__ == "__main__":
	app.run(debug=True)
