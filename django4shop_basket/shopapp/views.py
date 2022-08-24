from django.shortcuts import render

# Create your views here.
def mainFunc(request):
    return render(request, 'main.html')

def page1Func(request): 
    return render(request, 'page1.html')

def page2Func(request):
    return render(request, 'page2.html')

def cartFunc(request):
    # 신청한 제품을 세션에 담기
    name = request.POST['name'] 
    price = request.POST['price']
    # print(name, price)
    product = {'name':name, 'price':price}
    
    productList = []
    
    if 'shop' in request.session:
        productList = request.session['shop']
        productList.append(product)
        request.session['shop'] = productList
    else:
        productList.append(product)
        request.session['shop'] = productList
        
    print(productList)
    
    context = {}
    context['products'] = request.session['shop']
    return render(request, 'cart.html', context)
    
def buyFunc(request):
    if 'shop' in request.session:
        productList = request.session['shop']
        total = 0
        for p in productList:
            total += int(p['price'])
        print('총계 : ', total) 
        request.session.clear()   # del request.sesesion['shop']
        
    return render(request, 'buy.html', {'total':total})



