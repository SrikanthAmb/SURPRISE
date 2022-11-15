from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

ridge_model=pickle.load(open('ridge_model.pkl','rb'))

@app.route('/',methods=['GET'])

def Home():
    return render_template('real_index.html')



@app.route("/predict", methods=['POST'])
def predict():
    
   int_features=[int(x) for x in request.form.values()]
        
   features=[np.array(int_features)]
        
   prediction=ridge_model.predict(features)
                                                                         
        
   return render_template('real_index.html',prediction_text = "You Can purchase this House at {}".format(prediction))
    

if __name__=="__main__":
    app.run(debug=True)