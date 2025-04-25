from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('D:/Compozent-master/Machine Learning Project/breast.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/predict',methods=['POST']) 
def predict():
    if request.method == "POST":
        data = request.form
        mean_radius = float(data.get("mean_radius"))
        mean_texture = float(data.get("mean_texture"))
        mean_perimeter = float(data.get("mean_perimeter"))
        mean_area = float(data.get("mean_area"))
        mean_smoothness = float(data.get("mean_smoothness"))

        user_input = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
        model_output = model.predict(user_input)
        output_user = ''
        if model_output == 0:
            output_user = "Not diagnosed with breast-cancer"
        else:
            output_user = "Diagnosed with breast-cancer"
        return render_template('index.html', diagnosed=output_user)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
