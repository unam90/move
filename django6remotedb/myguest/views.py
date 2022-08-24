from django.shortcuts import render, redirect
from myguest.models import Guest
from datetime import datetime
from django.utils import timezone
from django.http.response import HttpResponseRedirect

# Create your views here.
def MainFunc(request):
    return render(request, 'main.html')  # urls를 만나지 않음

def SelectFunc(request):
   
    
    print(Guest.objects.filter(title__contains='안녕')) # 안녕이라는 글자가 있느냐 없느냐
    print(Guest.objects.get(id=1))
    print(Guest.objects.filter(id=1))
    print(Guest.objects.filter(title='연습'))
    
    # 정렬
    gdata = Guest.objects.all()  # 이것도 sql문인데 method로 만들어 놓은 것 뿐이다.
    # gdata = Guest.objects.all().order_by('title')  # 제목별 오름차순
    # gdata = Guest.objects.all().order_by('-title') # 제목별 내림차순
    # gdata = Guest.objects.all().order_by('-title', 'id') 
    # gdata = Guest.objects.all().order_by('-id')[0:2] # id별 내림차순 / 0번째와 1번째만 나옴(슬라이싱)
    
    return render(request, 'list.html', {'gdata':gdata})


def InsertFunc(request):
    return render(request, 'insert.html')

def InsertOkFunc(request):
    if request.method == 'POST':
        # print(request.POST.get('title'))
        Guest(
            title = request.POST['title'],
            content = request.POST['content'],
            # regdate = datetime.now()
            regdate = timezone.now()
        ).save()
    
    # return HttpResponseRedirect('/guest/select')  # 추가 후 목록보기. redirect 방식으로 요청
    return redirect('/guest/select')  # 두가지 방법 다 상관없음