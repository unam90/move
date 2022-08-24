from django.shortcuts import render
from app_jikwon.models import Jikwon
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

# Create your views here.
def mainFunc(request):
    return render(request, 'main.html')

def showFunc(request):
    # https://brownbears.tistory.com/101
    # 형식 : table1.objects.extra(tables=['table2'], where=['table2.id=group.num'])

    data = Jikwon.objects.extra(select = {'buser_name':'buser_name'}, 
                tables=['Buser'], 
                where=['Buser.buser_no=Jikwon.buser_num']).values\
                ('jikwon_no', 'jikwon_name', 'buser_name', 'jikwon_jik', 'jikwon_pay', 'jikwon_ibsail', 'jikwon_gen').\
                order_by('buser_name', 'jikwon_name')
               
    # values()로 부분추출 하지말고 필요한 요소만 dict에 담아서 추출
    pd.set_option('display.max_columns', 500) # 칼럼 모두 보기 옵션
    df = pd.DataFrame(data) 
    df.columns=['부서명', '사번', '직원명', '직급', '연봉', '입사', '성별'] 
    
    period = []
    for i in data.values('jikwon_ibsail'):
        period.append((date.today()).year - (i['jikwon_ibsail']).year)
    
    df['근무년수'] = period
    df.sort_values(by=['부서명','직원명'])
    del df['입사']
    # print(df)
   
    group = df['연봉'].groupby(df['부서명'])
    detail = {'sum':group.sum(), 'avg':group.mean()}
    
    group_j = df['연봉'].groupby(df['직급'])
    detail_j = {'sum':group_j.sum(), 'avg':group_j.mean()}
    
    result = group.agg(['sum','mean'])
    result.plot(kind='bar')
    plt.title("부서별 급여합, 급여평균")
    plt.ylabel("연봉")
    fig = plt.gcf()
    fig.savefig('django10_ex/app_jikwon/static/images/jik.png')
    
    ctab = pd.crosstab(df['성별'], df['직급'], margins=True)
    
    return render(request, 'show.html', 
                  {'df':df.to_html(), 
                   'detail':detail, 
                   'detail_j':detail_j, 
                   'ctab':ctab.to_html()})