from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/',methods=['GET'])

def Home():
    return render_template('real_index.html')



@app.route("/predict", methods=['POST'])
def predict():
    
   int_features=[int(x) for x in request.form.values()]
        
   features=[np.array(int_features)]
        
   prediction=model.predict(features)

   prediction=int(prediction[0])    

                                                                     
        
   return render_template('real_index.html',prediction_text = 'You can purchase this House at 'u"\u20B9" ' {}'.format(prediction))
    

if __name__=="__main__":
    app.run(debug=True)