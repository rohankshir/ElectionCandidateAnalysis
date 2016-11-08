import os
from flask import Flask,render_template, request,json
from  predictor import Predictor


app = Flask(__name__)
predictor = Predictor('classifier.pkl', 'lexical_vocab.pkl', 'pos_vocab.pkl')

@app.route('/')
def query():
    return render_template('query.html')

@app.route('/queryModel', methods=['GET','POST'])
def queryModel():

    query =  request.form['query']
    print query    
    y, y_prob, relevant_feats =  predictor.predict(query)
    
    return json.dumps({'status':'OK',
                       'prediction':y,
                       'probabilities':y_prob.tolist(),
                       'relevant_words':relevant_feats
                   })

if __name__=="__main__":
    app.run()
