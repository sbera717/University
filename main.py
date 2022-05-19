from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

file=open('covid.pkl','rb')
clf=pickle.load(file)
file.close()


@app.route('/')
def index():
   return render_template('index.html')
@app.route('/about')
def ab():
   return render_template('about.html')

@app.route('/test',methods=['GET','POST'])
def hello_world():
    if request.method=='POST': 
        myDict=request.form
        has_cough=int(myDict['cough'])
        has_fever=int(myDict['fever'])
        sorethroat=int(myDict['sore_throat'])
        Short_breadth=int(myDict['shortness_of_breath'])
        Taste=int(myDict['loss_of_taste_or_smell'])
        age_60=int(myDict['age_60_and_above'])
        Gen=int(myDict['gender'])
        inputFeatures=[has_cough,has_fever,sorethroat,Short_breadth,Taste,age_60,Gen]
        infProb=clf.predict_proba([inputFeatures])[0][1]
        res = "{:.2f}".format(infProb*100) 
        print(res)

        return render_template('show.html',inf=res)
    return render_template('test.html')
   

if __name__ == "__main__":
    app.run(debug=True)
