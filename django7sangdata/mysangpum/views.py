from django.shortcuts import render
import MySQLdb
from mysangpum.models import Sangdata
from django.http.response import HttpResponseRedirect
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage


# Create your views here.
def MainFunc(request):
    return render(request, 'main.html')

def ListFunc(request):
    """
    sql = "select * from sangdata"
    conn = MySQLdb.connect(**config)
    cursor.execute(sql)
    datas = cursor.fetchall()        # return type tuple
    """
    #페이징 처리 안한 경우
    # datas = Sangdata.objects.all()  # orm(object relational mapping)을 사용 / return type QuerySet
    # return render(request, 'list.html', {'sangpums':datas})
    
    # 페이징 처리 한 경우 
    datas = Sangdata.objects.all().order_by('-code')
    paginator = Paginator(datas, 5)  # 페이지 당 5행 씩 출력
    
    try:
        page = request.GET.get('page')
    except:
        page = 1    
        
    try:
        data = paginator.page(page)
    except PageNotAnInteger:
        data = paginator.page(1)
    except EmptyPage:
        data = paginator.page(paginator.num_pages())
        
    # 개별 페이지 표시용
    allpage = range(paginator.num_pages + 1)
    
    return render(request, 'list2.html', {'sangpums':data, 'allpage':allpage})
    
def InsertFunc(request):
    return render(request, 'insert.html')

def InsertOkFunc(request):
    if request.method == 'POST':
        # code = request.POST.get("code")
        # print(code)
        # 새로운 상품 code가 중복되는 지 검사 후 insert 진행
        try:
            Sangdata.objects.get(code=request.POST.get("code")) # 입력한 코드번호를 db에서 찾아보고 있으면,
            return render(request, 'insert.html', {'msg':'이미 등록된 번호입니다.'})
        except Exception as e:
            Sangdata(
                code = request.POST.get("code"),
                sang = request.POST.get("sang"),
                su = request.POST.get("su"),
                dan = request.POST.get("dan"),
            ).save()
        
        return HttpResponseRedirect('/sangpum/list')  # 추가 후 목록 보기
        
def UpdateFunc(request):
    data = Sangdata.objects.get(code=request.GET.get('code'))
    return render(request, 'update.html', {'sang_one':data})

def UpdateOkFunc(request):
    if request.method == 'POST':
        upRec = Sangdata.objects.get(code=request.POST.get("code"))
        upRec.code = request.POST.get("code")
        upRec.sang = request.POST.get("sang")
        upRec.su = request.POST.get("su")
        upRec.dan = request.POST.get("dan")
        upRec.save()
    
    return HttpResponseRedirect('/sangpum/list')  # 수정 후 목록 보기
        

def DeleteFunc(request):
    delRec = Sangdata.objects.get(code=request.GET.get("code"))
    delRec.delete()
    
    return HttpResponseRedirect('/sangpum/list')  # 삭제 후 목록 보기