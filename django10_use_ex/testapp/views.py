from django.shortcuts import render
from testapp.models import Jikwon
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

# Create your views here.
def mainFunc(request):
    return render(request, 'main.html')

def showFunc(request):
    # https://brownbears.tistory.com/101
    # 형식 : tabl1.objects.extra(tables=['table2'], where=['table2'.id=group.num'])
    data = Jikwon.objects.extra(select={'buser_name':'buser_name'},
                                 tables=['Buser'],
                                 where=['Buser.buser_no=Jikwon.buser_num']).values\
                                 ('jikwon_no', 'jikwon_name','buser_name','jikwon_jik','jikwon_pay','jikwon_ibsail', 'jikwon_gen').\
                                 order_by('buser_name', 'jikwon_name')
                                 
                
                                 
    
