from django.shortcuts import render
from django.http import HttpResponse
from hello import models
# Create your views here.

user_list=[
    {"user":"jack","pwd":"abc"},
    {"user":"tom","pwd":"ABC"},
]

def hello(request):
    if request.method=="GET":
        username=request.GET.get("username",None)
        password=request.GET.get("password",None)

        if username!=None and password!=None:
            models.UserInfo.objects.create(user=username, pwd=password)
    user_list=models.UserInfo.objects.all()
    return render(request,'hi.html',{"data":user_list})