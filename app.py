from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from werkzeug import secure_filename

model=load_model('model.h5')
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
    return render_template('index.html')
  

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        target = os.path.join(APP_ROOT, 'static/images/')
        f = request.files['file']
        data = os.path.join(target, "query.jpg")
        test_image = image.load_img(f.filename, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        with graph.as_default():
            y = model.predict_classes(test_image)
        d={0:'CercosporaleafspotGrayleafspot',1:'Commonrust',2:'NorthernLeafBlight',3:'healthy'}
        dc={0: 'Corn (maize)', 1: 'Corn (maize)', 2: 'Corn (maize)',3: 'Corn (maize)'}
        crop=dc[y[0]]
        dis=d[y[0]]
    return render_template("prediction.html",result = {'Crop':crop,'Disease':dis})
if __name__ == "__main__":
    app.run(debug=False)



