from django.shortcuts import render, redirect
from mysurvey.models import Survey
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

# Create your views here.
def surveyMain(request):
    return render(request, 'main.html')

def surveyView(request):
    return render(request, 'survey.html')

def surveyProcess(request):
    insertData(request)
    return redirect('/coffee/surveyshow')  # 자료 입력 후 분석 결과 보기 요청

    
def insertData(request):  # 테이블에 자료 추가
    if request.method == 'POST':
        # print(request.POST.get('gender'))
        # print(request.POST.get('age'))
        # print(request.POST.get('co_survey'))
        Survey(
            gender = request.POST.get('gender'),
            age = request.POST.get('age'),
            co_survey = request.POST.get('co_survey')
        ).save()

def dataAnalysis(request):
    rdata = list(Survey.objects.all().values())
    # print(rdata)
    df = pd.DataFrame(rdata)
    df.dropna()  # 결측치 제거
    print(df)
    
    # dummy 변수 처리 후 해도 됨. 남=0, 여=1 ...
    
    
    ctab = pd.crosstab(index=df['gender'], columns=df['co_survey'])
    # print(ctab) 
    
    # 이원 카이제곱 검정
    chi, pv, _, _ = stats.chi2_contingency(observed=ctab)
    print('chi, pv:', chi, pv)
    
    if pv > 0.05:
        result = 'p값이 {0} > 0.05 이므로<br>성별과 커피브랜드의 선호도는 관계가 없다.<br><b>귀무가설 채택</b>'.format(pv)
    else:
        result = 'p값이 {0} < 0.05 이므로<br>성별과 커피브랜드의 선호도는 관계가 있다.<br><b>대립가설 채택</b>'.format(pv)
        
    count = len(df)
    
    # 커피브랜드별 선호 건수에 대한 차트(세로막대)를 출력하시오
    fig = plt.gcf()
    gender_group = df.groupby(['co_survey'])['rnum'].count()
    # print(gender_group)
    gender_group.plot.bar(subplots=True, color=['cyan','green'], width=0.5)
    plt.xlabel('커피 브랜드명')
    plt.title('커피브랜드별 선호 건수')
    plt.grid()
    fig.savefig('django11_chi/mysurvey/static/images/coffee.png')
    
    return render(request, 'list.html', {'ctab':ctab.to_html(), 'result':result, 'count':count})
    
    
    
    
    
    
    
    