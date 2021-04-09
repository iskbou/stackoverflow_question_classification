from joblib import load
import numpy as np
import pandas as pd

pa=load('C:/Users/Niskiwork/Documents/Silexia/GitHub/stackoverflow_question_classification/passiveagressive.joblib')
proc=load('C:/Users/Niskiwork/Documents/OpenClassroom/Projet 5 - StackExhange Questions tag/processing.joblib')
labenc=load('C:/Users/Niskiwork/Documents/OpenClassroom/Projet 5 - StackExhange Questions tag/label.joblib')
test=pd.DataFrame(columns=['year','Title','Body_text'])
test['Title']=["How can I check if a Pandas dataframe's index is sorted"]
test['Body_text']=['I have a vanilla pandas dataframe with an index. I need to check if the index is sorted. Preferably without sorting it again.\ne.g. I can test an index to see if it is unique by index.is_unique() is there a similar way for testing sorted?\n']
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