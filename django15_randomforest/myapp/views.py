from django.shortcuts import render
import joblib

# Create your views here.

def mainFunc(request):
    return render(request, 'main.html')

def showFunc(request):
    pass
