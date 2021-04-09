from joblib import load
import numpy as np
import pandas as pd

pa=load('C:/Users/Niskiwork/Documents/Silexia/GitHub/stackoverflow_question_classification/passiveagressive.joblib')
proc=load('C:/Users/Niskiwork/Documents/OpenClassroom/Projet 5 - StackExhange Questions tag/processing.joblib')
labenc=load('C:/Users/Niskiwork/Documents/OpenClassroom/Projet 5 - StackExhange Questions tag/label.joblib')
test=pd.DataFrame(columns=['year','Title','Body_text'])
test['Title']=['What does "a" stand for in font: 0/0 a;']
test['Body_text']=["I was referring a video tutorial where the designer used font: 0/0 a; for image replacement, so I get that 0 is the font-size, another 0 is the line-height but designer skips the a part just by saying that's an hack.\nSo what does that a exactly do there?\n"]
test['year']=[2021]


def proba_label(model,X,n_class):
    try:
            x=pd.DataFrame(model.predict_proba(X))
    except:
            x=pd.DataFrame(model.decision_function(X))
    x.columns=labenc.inverse_transform(x.columns)
    z=pd.DataFrame(index=x.index,columns=[str(n_class)+' most likely tags'])
    z[str(n_class)+' most likely tags']=[
        {k:v for k,v in x.transpose()[i].sort_values(ascending=False)[:n_class].items()} for i in x.index
    ]
    return (x,z[str(n_class)+' most likely tags'])