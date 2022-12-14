from django.shortcuts import render
from mysangpum.models import Sangdata
from django.http.response import HttpResponse
import json

# Create your views here.
def mainFunc(request):
    return render(request, 'main.html')


def listFunc(request):
    return render(request, 'list.html')

def listDbFunc(request):
    sdata = Sangdata.objects.all()
    
    datas = []
    for s in sdata:
        dic = {'code':s.code, 'sang':s.sang, 'su':s.su, 'dan':s.dan}
        datas.append(dic)
        
    # print(datas)
    return HttpResponse(json.dumps(datas), content_type='application/json')



