from django.shortcuts import render
from myapp.models import Article

# Create your views here.
def main(request):
    return render(request, 'main.html')


def showdb(request):
    # 1. sql문을 직접 사용해서 html에 전달
    # 2. Django의 orm 기능을 사용 (권장)
    datas = Article.objects.all()
    print(datas, type(datas))   # QuerySet 타입
    print(datas[0].name)
    
    return render(request, 'list.html', {'articles':datas})
