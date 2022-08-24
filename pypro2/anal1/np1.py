# numpy : ndarray 지원. 파이썬의 데이터 관련 라이브러리 전체 생태계의 핵심을 지원.

# 직접 분산, 표준편차를 구하고, numpy의 함수와 비교

grades = [1, 3, -2, 4]

def show_grades(grades):
    for g in grades:
        print(g, end= ' ')
        
show_grades(grades)

def grades_sum(grades):
    tot = 0 
    for g in grades:
        tot += g
    return tot
print()
print('합은 ', grades_sum(grades))

def grades_mean(grades):
    tot = grades_sum(grades)
    m = tot / len(grades)
    return m 

print('평균은 ', grades_mean(grades))

# 분산 : 평균값을 기준으로 다른 값들의 흩어진 정도
def grades_variance(grades):
    m = grades_mean(grades)
    vari = 0
    for num in grades:
        vari += (num - m) ** 2  # 편차 제곱의 합
    return vari / len(grades)  # 모집단으로 계산
    
print('분산은 ', grades_variance(grades))

# 표준편차 : 분산에 루트를 씌운 값
def grades_std(grades):
    return grades_variance(grades) ** 0.5  

print('표준편차는 ', grades_std(grades))


import numpy
print(numpy.__version__)  # numpy 버전 확인
print('합은 ', numpy.sum(grades))
print('평균은 ', numpy.mean(grades))
print('분산은 ', numpy.var(grades))
print('표준편차는 ', numpy.std(grades))

print(numpy.average(grades))
print(numpy.mean(grades))




    
