from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
from my_jikwon.models import Jikwon
plt.rc('font', family='malgun gothic')

# Create your views here.
def mainFunc(request):
    return render(request, 'main.html')

def showFunc(request):
    jikwons = Jikwon.objects.all().values()
    print(jikwons)
    df = pd.DataFrame.from_records(jikwons)
    df.columns=['사번','직원명','부서','직급','연봉','입사','성별','평점']
    print(df.head(2))
    
    # 직급별 연봉합, 연봉평균 구하기
    jik_group = df['연봉'].groupby(df['직급'])
    # print(jik_group.sum())
    jik_group_detail = {'sum':jik_group.sum(), 'avg':jik_group.mean()}
    
    df2 = pd.DataFrame(jik_group_detail)
    df2.columns = ['연봉합','연봉평균']
    print(df2)
    
    ctab = pd.crosstab(df['직급'], df['평점'])  # 직급별 평점별 건수 빈도표
    print(ctab)
    
    # 시각화 : 직급별 연봉합, 평균
    jik_result = jik_group.agg(['sum', 'mean'])
    # print('jik_result:', jik_result)
    jik_result.plot(kind = 'barh')
    plt.title('직급별 연봉합과 평균')
    plt.xlabel('연봉')
    fig = plt.gcf()
    fig.savefig('django9_jikwon/my_jikwon/static/images/jik.png')
    
    ypays = list(df['연봉'])
    names = list(df['사번'])
    print('ypays : ', ypays)
    print('names : ', names)                        
    
    return render(request, 
                  'show.html', 
                  {'datas':df.to_html(index=False, border=0),
                   'jik_group':jik_group_detail,
                   'jik_group2':df2.to_html(index=False),
                   'ctab':ctab.to_html(),
                   'ypays':ypays,
                   'names':names
                   })
    
    
    
