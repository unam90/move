# 문제1) 1 ~ 100 사이 정수 중 3의 배수이나 2의 배수가 아닌 수들의 합
i = 0
hap = 0

while i < 100:
    if i % 3 == 0 and i % 2 != 0:
        #print(i, end = ' ')
        hap += i 
    i += 1
print('합은 ' + str(hap))


# 문제2) 2~5까지의 구구단
i = 2
while i <= 5:
    j = 1
    while j <= 9:
        print(i, "X", j, "=", i*j)
        j = j + 1
    i += 1
print()


# 문제3) -1, 3, -5, 7, -9, 11 ~ 99 까지의 모두에 대한 합을 출력  
i = 1
cnt = 1
hap = 0

while i < 100:
    if cnt % 2 == 0: # 짝수 지점
        #print(i)
        #합을 구해야 하니까
        hap += i
    
    else:            # 홀수 지점(음수)
        k = i * -1   # 음수로 변경
        #print(k)
        hap += k 
        
    cnt += 1
    i += 2           # 1,3,5,7,9...
    
    print('합은:', hap)
    

# 문제4) 1~1000 사이의 소수 갯수 출력
i = 2
sosu = []

while i <= 1000:
    count = 0
    j=1
    while j <= i:
        if i % j == 0:
            count+=1
        j+=1
    if count==2:
        #print(i, end=' ')
        sosu.append(i)
    i+=1
print()
print(len(sosu))        



