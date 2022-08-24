from django.urls import path
from myguest import views

urlpatterns = [
    path('select', views.SelectFunc), 
    path('insert', views.InsertFunc),
    path('insertok', views.InsertOkFunc), 

]