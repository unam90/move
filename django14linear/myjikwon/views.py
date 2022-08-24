from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
from myjikwon.models import Jikwon
from datetime import datetime
import joblib
from django.http.response import HttpResponse, JsonResponse

# Create your views here.
flag = False

def mainFunc(request):
    global flag
    if flag == False:
        makeModel()
        flag = True
        
    return render(request, 'show.html')

def makeModel():
    # jikwon 테이블에서 근무년수에 대한 연봉을 이용하여 회귀분석 모델을 작성
    datas = Jikwon.objects.values('jikwon_ibsail', 'jikwon_pay', 'jikwon_jik').all()
    jikdf = pd.DataFrame.from_records(datas)
    # print(jikdf.head(3), len(jikdf))
    
    # 근무년수 구하기
    for i in range(len(jikdf['jikwon_ibsail'])):
        jikdf['jikwon_ibsail'][i] = \
            int((datetime.now().date() - jikdf['jikwon_ibsail'][i]).days / 365)
    jikdf.columns = ['근무년수', '연봉', '직급']
    # print(jikdf.head(3))
    #   근무년수    연봉  직급
    # 0   13  9900  이사
    # 1   12  8800  부장
    # 2   12  7900  과장
    
    # 과적합 방지를 목적으로 data를 train / test로 분리해서 train으로 학습, test로 모델을 검정
    train_set, test_set = train_test_split(jikdf, test_size = 0.2, random_state=12)  # 8:2
    print(train_set.shape, test_set.shape)  # (24, 3) (6, 3)
    
    # 모델 생성 : 근무년수(feature(독립변수)), 연봉(label(종속변수))
    model_lm = LinearRegression().fit(X=train_set.iloc[:,[0]], y=train_set.iloc[:,[1]])
    
    # 성능 확인
    test_pred = model_lm.predict(test_set.iloc[:, [0]])
    print('연봉 예측값:', test_pred[:5].flatten())  
    print('연봉 실제값:', test_set.iloc[:, [1]][:5].values.flatten())
    
    # 선형회귀모델 성능을 파악하기 위한 방법
    global r2s
    r2s = r2_score(test_set.iloc[:, [1]], test_pred)
    print('결정계수(설명력, r2_score) : ', r2s)  # 0.48881
    
    
    # 모델 저장   
    joblib.dump(model_lm, 'django14linear/myjikwon/static/django14.model')
    
    # 직급별 연봉 평균
    global pay_jik 
    pay_jik = jikdf.groupby('직급').mean().round(1)
    print('pay_jik:', pay_jik)
    
    
@csrf_exempt    # csrf를 적용하고 싶지 않은 경우 적는다.
def predictFunc(request):
    year = request.POST['year']
    # print('year :', year)
    new_year = pd.DataFrame({'year':[year]})
    
    # 모델을 읽어 입력된 년도에 해당하는 연봉을 예측하여 클라이언트로 반환
    model = joblib.load('django14linear/myjikwon/static/django14.model')
    
    new_pred = round(model.predict(new_year)[0][0], 2)
    print('new_pred:', new_pred)
    
    context = {'new_pred':new_pred, 'r2s':r2s, 'pay_jik':pay_jik.to_html()}
    # return HttpResponse(json.dumps(context), content_type='application/json')
    return JsonResponse(context)  # 위와 동일한 기능

    