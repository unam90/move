# 특성공학 중 feature extraction (차원축소 : PCA - 주성분 분석)
# 주성분 분석(Principal component analysis; PCA)은 고차원의 데이터를 저차원의 데이터로
# 환원시키는 기법을 말한다. 이 때 서로 연관 가능성이 있는 고차원 공간의 표본들을 선형 연관성이 
# 없는 저차원 공간(주성분)의 표본으로 변환하기 위해 직교 변환을 사용한다.
# PCA는 선형대수 관점으로 볼 때 입력 데이터의 공분산 행렬을 고유값 부해하고
# 이렇게 구한 고유벡터에 입력데이터를 선형 변환 하는 것이다.

# 참고 : PCA 진행 순서
# 1) 입력 데이터세트의 공분산 행렬을 생성한다.
# 2) 공분산 행렬의 고유벡터(행렬의 방향은 그대로, 크기만 변하는 벡터)와 고유값을 계산한다.
# 3) 고유값이 큰 순으로 k(변환 차수)개 만큼 고유벡터를 추출한다.
# 4) 추출된 고유벡터를 이용하여 새롭게 입력 데이터를 변환한다.

# iris dataset 중 sepal(꽃받침) 열의 너비와 길이, 두 개를 하나의 열로 차원 축소하기
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='malgun gothic')

iris = load_iris()
n = 10
x = iris.data[:n, :2]  # sepal
print('차원 축소 전 x:', x, x.shape)
print(x.T)

"""
# 시각화
plt.plot(x.T, 'o:')
plt.xticks(range(2), ['길이', '너비'])
plt.xlim(-0.2, 2)
plt.ylim(2.5, 6)
plt.xlabel('특성의 종류')
plt.ylabel('특성값')
plt.grid(True)
plt.legend(['표본{}'.format(i + 1) for i in range(10)])
plt.show()
# 두개의 열 움직임이 유사
"""

# 두 열 값에 대해 변동성이 가장 큰 방향의 축 생성하고 PCA를 진행
df = pd.DataFrame(x)
print(df)

# 시각화
""" 
ax = sns.scatterplot(0, 1, data=df, marker='s', s=100, color=['b'])
# text
for i in range(n):
    ax.text(x[i, 0] - 0.05, x[i, 1] + 0.03, '표본{}'.format(i + 1))
plt.xlabel('길이')
plt.ylabel('너비')
plt.title('아이리스 꽃받침 자료')
plt.show()
"""

# PCA
pca1 = PCA(n_components=1)  # 차원 축소 입력 인수
x_row = pca1.fit_transform(x)  # 비지도 학습이므로 target(y)은 지정하지 않음
print('x_row:', x_row, ' ', x_row.shape)  # 오리지날 데이터의 근사 데이터의 집합

x2 = pca1.inverse_transform(x_row)  # 원복
print('원복:', x2, ' ', x2.shape)

print(x_row[0])  # [0.30270263]
print(x2[0, :])  # [5.06676112 3.53108532] 원래 값 : 5.1 3.5

# 시각화
ax = sns.scatterplot(0, 1, data=df, marker='s', s=100, color=['b'])
# text
for i in range(n):
    ax.text(x[i, 0] - 0.05, x[i, 1] + 0.03, '표본{}'.format(i + 1))
    plt.plot([x[i, 0], x2[i, 0]], [x[i, 1], x2[i, 1]], 'k--')

plt.plot(x2[:, 0], x2[:, 1], 'o-', color='g', markersize=10)  # 근사 행렬값 표시

plt.xlabel('길이')
plt.ylabel('너비')
plt.title('아이리스 꽃받침 자료')
plt.axis('equal')
plt.show()

print('--------------------')
# iris 자료의 열 4개를 2개로 차원 축소 
x = iris.data
print(x[:2])
pca2 = PCA(n_components = 2)
x_row2 = pca2.fit_transform(x)
print('x_row2:', x_row2[:2], ' ', x_row2.shape)
print(pca2.explained_variance_ratio_)  # PCA 변동성 비율
# [0.92461872 0.05306648] : [제1주성분, 제2주성분] 
# 첫번째 반환 요소값이 전체 변동성에 92.5% 정도를 설명하고 있다.
# 2개의 요소만 있으면 원본데이터의 변동성을 거의 다 설명할 수 있다는 뜻.
x4 = pca2.inverse_transform(x_row2)
print()
print('최초 자료:', x[0])           # [5.1 3.5 1.4 0.2]
print('차원 축소 자료:', x_row2[0])  # [-2.68412  0.31939]
print('차원 복귀 자료:', x4[0, :])   # [5.0830 3.5174 1.4032 0.2135]

print()
iris1 = pd.DataFrame(x, columns=['꽃받침길이', '꽃받침너비', '꽃잎길이', '꽃잎너비'])
iris2 = pd.DataFrame(x_row2, columns=['변수1', '변수2'])
print(iris1.head(3))
print(iris2.head(3))




