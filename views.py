from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
# Create your views here.
from django.http import HttpResponseRedirect,HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
import pandas as pd
from . import etiquetage
import json,csv,time
from joblib import load
import os
from django.views.decorators.csrf import csrf_exempt

cwd = os.getcwd()


# Create your views here.
#s1 =open(pa1.txt,'r')
#s2 =open(pa2.txt,'r')
#pa=s1+s2
#s1.close()
#s2.close()
pa=load(cwd+'/classification_stack/passiveagressive.joblib')
proc=load(cwd+'/classification_stack/processing.joblib')
#labenc=load(cwd+'/classification_stack/label.joblib')
#sgd=load(cwd+'/classification_stack/modified_huber_sgd.joblib')
mullabenc=load(cwd+'/classification_stack/mullabel.joblib')

n_classes=5

 
class TemplateView(generic.TemplateView):
    template_name = 'classification_stack/classification_demo.html'

@csrf_exempt    
def etiquette(request):
    titre = str(request.POST.get('titre'))
    corps=str(request.POST.get('corps'))
    year=int(time.ctime()[-4:])
    donnees=pd.DataFrame(columns=['year','Title','Body_text'])
    donnees['Title']=[titre]
    donnees['Body_text']=[corps]
    donnees['year']=[year]
#    resultat=etiquetage.proba_label(sgd,proc.transform(donnees),n_classes,labenc)
    resultat=mullabenc.inverse_transform(pa.predict(proc.transform(donnees)).reshape(1,-1))
#    resultat=etiquetage.proba_label(sgd,proc.transform(donnees),n_classes,labenc)
#    return JsonResponse({'tag':[i for i in resultat[0].keys()], 'valeurs':[i for i in resultat[0].values()]})
    return JsonResponse({'tag':[i for i in resultat[0]]})
    #dict(resultat)