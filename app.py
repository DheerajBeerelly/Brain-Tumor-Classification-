import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img
import pandas
import cv2
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = load_model('D:\Study\B.Tech\Semesters\SEM 8\Major Project\Code\mymodel.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
        
    img = Image.open(request.files['img'])  
    
    
    img = img.resize((128,128))
    #print(img)
    img = img.convert("RGB")

    
    x = np.array(img)
    #print(x.shape,end="\n")
    x = x.reshape((1,) + x.shape)
    
    res = model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    
    

    return render_template('index.html', prediction_text='{}'.format(names(classification)))

def names(number):
    if number==0:
        return 'Yes, Its a Tumor'
    else:
        return 'No, Its not a tumor'

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
