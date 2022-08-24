from django.shortcuts import render
from django.views.generic.base import TemplateView

# Create your views here.
def MainFunc(request):
    return render(request, 'index.html')
    

class MyCallView(TemplateView):
    template_name = 'callget.html'


def InsertFunc(request):
    # return render(request, 'insert.html')  # get,post를 구분하지 않을 때 처리 방법 
    
    # 같은 요청명에 대해 get, post를 구분해서 처리 가능
    if request.method == 'GET':
        return render(request, 'insert.html')
    elif request.method == 'POST':
        buser = request.POST.get('buser')
        irum = request.POST['irum']
        print(buser, irum)
        
        # buser, irum으로 뭔가를 하면 된다.
        msg1 = '부서 :' + buser
        msg2 = '직원 이름 :' + irum
        context = {'msg1': msg1, 'msg2': msg2}
        return render(request, 'show.html', context)
    else:
        print('요청 오류')


def InsertFuncOk(request):
    # buser = request.GET.get('buser')
    # irum = request.GET['irum']
    buser = request.POST.get('buser')
    irum = request.POST['irum']
    print(buser, irum)
    
    # buser, irum으로 뭔가를 하면 된다.
    msg1 = '부서 :' + buser
    msg2 = '직원 이름 :' + irum
    context = {'msg1': msg1, 'msg2': msg2}
    return render(request, 'show.html', context)

