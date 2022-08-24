from django.shortcuts import render
from django.http.response import HttpResponseRedirect

# Create your views here.
def mainFunc(request):
    return render(request, 'main.html')


def setOsFunc(request):
    # print(request.GET)
    if "favorite_os" in request.GET:  # get 방식의 요청값이 "favorite_os"인 경우 처리
        print(request.GET["favorite_os"])  
        # request.session['세션키']
        request.session['f_os'] = request.GET["favorite_os"]  # 세션 생성 
        return HttpResponseRedirect("/showos")  # redirect 방식 (클라이언트를 통해서 서버에 요청)
    else:
        return render(request, 'selectos.html')  # forward 방식 : 서버에서 서버 파일을 호출

def showOsFunc(request):
    # print('무사히 여기까지 도착')
    
    dict_context = {}  # 세션 자료를 html 파일에 전달할 목적
    
    if "f_os" in request.session:  # session 값 중에 "f_os" 유무
        print('유효시간:', request.session.get_expiry_age())
        dict_context['sel_os'] = request.session['f_os']
        dict_context['message'] = "당신이 선택한 운영체제는 %s"%request.session['f_os']
    else:
        dict_context['sel_os'] = None
        dict_context['message'] = "운영체제를 선택하지 않았네요"
    
    # 참고 : 특정 세션 삭제 request.session['키이름']
    # set_expiry(0) 하면 브라우저가 닫힐 때 세션이 해제됨
    
    request.session.set_expiry(5)  # 5초 동안 세션이 유효. 기본값은 30분
        
    return render(request, 'show.html', dict_context)
        
    
    