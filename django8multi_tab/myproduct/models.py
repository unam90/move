from django.db import models

# Create your models here.
class Maker(models.Model):
    mname = models.CharField(max_length=10)
    tel = models.CharField(max_length=30)
    addr = models.CharField(max_length=50)
    
    class Meta:
        ordering = ('-id',)  # id별 descending sort
        
    def __str__(self):
        return self.mname

class Product(models.Model):
    pname = models.CharField(max_length=10)
    price = models.IntegerField()
    maker_name = models.ForeignKey(Maker, on_delete=models.CASCADE)