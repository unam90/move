from django.urls import path
from gpapp import views

urlpatterns = [
    path('insert', views.InsertFunc),
    
   # path('insertok', views.InsertFuncOk), 
]