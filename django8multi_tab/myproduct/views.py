from django.shortcuts import render
from myproduct.models import Maker, Product
import pandas as pd
import MySQLdb
import pickle

config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'maria123',
    'database':'productdb',
    'port':3306,
    'charset':'utf8',
    'use_unicode':True
}


# Create your views here.

def Main(request):
    return render(request, 'main.html')

def List1(request):
    # 제조사 정보 출력
    # 1) Django ORM
    makers = Maker.objects.all()  # 반환값이 쿼리셋 타입
    print(type(makers))
    return render(request, 'list1.html', {'makers':makers})
    
    """
    # 2) Django ORM의 결과를 pd.DataFrame으로 저장 후 전송
    df = pd.DataFrame(list(Maker.objects.all().values()))
    # df = pd.DataFrame(list(Maker.objects.all().values('mname','tel')))
    print(df)
    # return render(request, 'list1_1.html', {'makers':df.to_html(index=False)})

    # 3) SQL문 사용
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = "select * from myproduct_maker"
    cursor.execute(sql)
    rows = cursor.fetchall()  # <class 'tuple'>
    print(type(rows))
    # df = pd.DataFrame(rows)
    # return render(request, 'list1_1.html', {'makers':df.to_html(index=False)})

    # 4) SQL문 사용 - pandas 기능
    conn = MySQLdb.connect(**config)
    df = pd.read_sql("select * from myproduct_maker", conn)
    return render(request, 'list1_1.html', {'makers':df.to_html(index=False)})
    """
from django.db.models.aggregates import Count, Sum, Avg, StdDev, Variance
    
def List2(request):
    # 제품 정보 출력
    products = Product.objects.all()
    pcount = len(products)
    
    # ORM연습
    print(products.values_list())
    print(products.aggregate(Count('price')))
    print(products.aggregate(Avg('price')))
    
    imsi = products.values('pname').annotate(Avg('price'))
    print(imsi)
    for r in imsi:
        print(r)
    
    return render(request, 'list2.html', {'products':products, 'pcount':pcount})


def List3(request):
    mid = request.GET.get('id')
    products = Product.objects.filter(maker_name=mid)
    pcount = len(products)
    return render(request, 'list2.html', {'products':products, 'pcount':pcount})
    
    
    
    