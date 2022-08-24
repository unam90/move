from django.db import models

# Create your models here.
class Guest(models.Model):
    # myno = models.AutoField(auto_created=True, primary_key=True)
    title = models.CharField(max_length=50)
    content = models.TextField()
    regdate = models.DateTimeField()
    
    # 정렬하기 방법 2
    class Meta:
        # ordering = ('title', 'id')
        ordering = ('-id',)
        
        