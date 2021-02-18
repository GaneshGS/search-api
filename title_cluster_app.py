from flask import Flask, request, render_template
import spacy
import en_core_web_sm
import pickle as pkl

app = Flask(__name__)
#title_model = pkl.load(open('kmeans.pkl', 'rb'))
#nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('voice_navigation')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():#For rendering results on HTML GUI
    output = []
    title = request.form["Title"]
    vector_emb = nlp(str(title))
    for ent in vector_emb.ents:
        output.append(ent.text)
        output.append(ent.label_)
    prediction = output
    #prediction = title_model.predict([vector_emb.vector])
    return render_template('index.html', title=title, Cluster_label=prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5000)


'''
#from flask_caching import Cache
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

text = ['Director of Business Development', 'GENERAL WAREHOUSE WORKER', 'Acute Dialysis Nurse', 'Truck Driver', 'Remote Patient Monitoring Specialist',
        'Field Service Coordinator', 'Field Service Technician', 'Heavy Equipment Operator', ]
vector_emb = embed(text)

print(title_model.predict(vector_emb))

#cache = Cache(app, config={"CACHE_TYPE": "simple"})
#cache.init_app(app)
#@cache.cached(timeout=10)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
'''
