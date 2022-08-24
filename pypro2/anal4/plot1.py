# 시각화 : matplotlib 라이브러리를 사용
# figure : 그래프(차트)가 그려지는 영역
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family = 'malgun gothic')    # font 변경
plt.rcParams['axes.unicode_minus'] = False  # minus기호 깨짐 방지

"""
x = ['서울', '인천','수원']  
# list type 숫자가 매겨져 있음. x축과 y축에는 숫자가 들어가야한다. set type은 순서가 없으므로 안됨
y = [5,3,7]
plt.plot(x, y)
plt.xlim([-1, 3])
plt.ylim([0, 10])
plt.yticks(list(range(0,11,3)))
plt.plot(x, y)
plt.show()
"""

"""
data = np.arange(1, 11, 2)
print(data)
plt.plot(data)
x = [0,1,2,3,4]
for a, b in zip(x, data):
    plt.text(a, b, str(b))
plt.show()
"""

"""
x = np.arange(10)
y = np.sin(x)
print(x, y)
# plt.plot(x, y, 'bo')
plt.plot(x, y, 'r-.', linewidth=3, markersize=12)  # 스타일을 줄 수 있다.
plt.show()
"""

"""
# hold : 하나의 figure 내에 plot을 복수로 표현 
x = np.arange(0, np.pi*3, 0.1)  # 증가치는 0.1
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.figure(figsize=(10,5))
plt.plot(x, y_sin, color ='r')  # 선그래프
plt.scatter(x, y_cos, color ='b')  # 산점도(산포도)
plt.xlabel('x축')
plt.ylabel('사인&코사인')
plt.legend(['sine','cosine'])
plt.show()
plt.xlabel('x축')
plt.plot(x, y_sin, color ='r')
"""

"""
# subplot : figure 영역을 여러개로 분할 
x = np.arange(10)
y = np.sin(x)

plt.subplot(2,1,1)  # 2행 1열 1번째 행
plt.plot(x)
plt.title('첫번째')

plt.subplot(2,1,2)  # 2행 1열 2번째 행
plt.plot(y)
plt.title('두번째')

plt.show()
"""

irum = ['a', 'b', 'c', 'd', 'e']
kor = [80,50,70,70,90]
eng = [60,20,80,70,50]
plt.plot(irum, kor, 'ro-')
plt.plot(irum, eng, 'bs--')
plt.ylim([0, 100])
plt.title('시험점수')  # 차트 제목
plt.legend(['국어', '영어'], loc=2)  # 범례 / loc(location)
plt.grid(True)

# 차트를 이미지로 저장
fig = plt.gcf()
plt.show()
fig.savefig('test.png')

# 이미지 파일 읽기
from matplotlib.pyplot import imread
img = imread('test.png')
plt.imshow(img)
plt.show()
