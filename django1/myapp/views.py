from django.shortcuts import render

# Create your views here.
def mainFunc(request):
    name = "홍길동"
    return render(request, 'main.html', {'msg':name})

def helloFunc(request):
    str = "<h1>출력을 위한 작업</h1>"
    return render(request, 'disp.html', {'str':str})