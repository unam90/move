from django.shortcuts import render
import json
from django.forms.fields import JSONString
from django.http.response import HttpResponse

lan = {
    'id': 111, 
    'name': '파이썬',
    'history':[
        {'date':'2022-8-3', 'exam':'basic'},
        {'date':'2022-8-3', 'exam':'django'},
    ]
}

def testFunc():
    print(type(lan))
    
    jsonString = json.dumps(lan)  # json encoding
    print(jsonString, type(jsonString))
    
    dic = json.loads(jsonString)  # json decoding
    print(dic, type(dic))
    
    
# Create your views here.
def indexFunc(request):
    testFunc()
    return render(request, 'abc.html')

import time

def func1(request):
    msg = request.GET.get('msg')
    msg = "nice " + msg
    print(msg)
    time.sleep(5)
    
    context = {'key':msg}
    return HttpResponse(json.dumps(context), content_type='application/json')  
    # content_type(mime type) : 클라이언트 브라우저한테 데이터타입의 해석방법을 알려주는 것

    
def func2(request):
    datas = [
        {'irum':'사오정', 'nai':22},
        {'irum':'삼오정', 'nai':33}
    ]
    return HttpResponse(json.dumps(datas), content_type='application/json')  


    