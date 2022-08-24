# 비계층적 군집분석 - K-means clustering
# 군집화는 아무런 정보가 없는 상태에서 데이터를 분류하는 방법이다. K-means Clustering 이란 데이터
# 분류 종류를 K개 라고 했을 때 입력한 데이터 중 임의로 선택된 K 개의 기준과 각 점들의 거리를 오차로
# 생각하고 각각의 점들은 거리가 가장 가까운 기준에 해당한다고 생각하는 것이다. 그리고 이제 각각 기준에
# 해당하는 점들 모두의 평균을 새로운 기준으로 갱신해 나가게 된다. 이렇게 해서 가장 적절한 중심점들을
# 찾는 것이다. 이렇게 학습을 반복하면 데이터를 분류할 수 있게 된다.

# 과정
# 1. 클러스터 수 즉, k값(군집 중심점) 결정
# 2. 데이터 공간에 클러스터 중심 k개 할당
# 3. 각 클러스터 중심을 해당 클러스터 데이터의 평균으로 조정
# 4. 클러스터 중심이 변하지 않을 때까지 반복

from sklearn.datasets import make_blobs  # 클러스터링 연습용 데이터를 제공
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# print(make_blobs)
x, y = make_blobs(n_samples=150, n_features=2, centers=3, 
                  cluster_std=0.5, shuffle=True, random_state=0)
print(x)
print(y)  # 군집작업에 의미없는 값이다.
print(x.shape)

# plt.scatter(x[:, 0], x[:, 1], s=30, c='gray', marker='o')
# plt.grid(True)
# plt.show()

print('K-means clustering')
init_centroid = 'random'
init_centroid = 'k-means++'  # default

kmodel = KMeans(n_clusters=3, init=init_centroid, random_state=0).fit(x)

pred = kmodel.fit_predict(x)
print('pred:', pred)

# 각 군집별 자료 보기 
print(x[pred==0], len(x[pred==0]))
print()
print(x[pred==1], len(x[pred==1]))
print()
print(x[pred==2], len(x[pred==2]))

print()
print('중심점(centroid) :', kmodel.cluster_centers_)

# 시각화
plt.scatter(x[pred==0,0], x[pred==0,1], s=50, c='red', marker='o', label='cluster1')
plt.scatter(x[pred==1,0], x[pred==1,1], s=50, c='green', marker='s', label='cluster2')
plt.scatter(x[pred==2,0], x[pred==2,1], s=50, c='blue', marker='v', label='cluster3')
plt.scatter(kmodel.cluster_centers_[:, 0], kmodel.cluster_centers_[:, 1], 
            s=80, c='black', marker='+', label='centroid')
plt.legend()
plt.grid(True)
plt.show()

# k값 얻기
# 1) 계층적 군집 분석의 결과로 k개 결정
# 2) elbow 기법 : 클러스터 내 SSE(오차제곱합)의 합이 최소가 되도록 클러스터의 중심을 결정해 나가는 방법

def elbow(x):
    sse = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', random_state=0).fit(x)
        sse.append(km.inertia_)
    
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('cluster num')
    plt.ylabel('SSE')
    plt.show()
    
elbow(x)

# 3) silhouette 기법 : 클러스터의 품질을 정량적으로 계산해 주는 방법
import numpy as np
from sklearn.metrics import silhouette_samples
from matplotlib import cm

# 데이터 X와 X를 임의의 클러스터 개수로 계산한 k-means 결과인 y_km을 인자로 받아 각 클러스터에 속하는 데이터의 실루엣 계수값을 수평 막대 그래프로 그려주는 함수를 작성함.
# y_km의 고유값을 멤버로 하는 numpy 배열을 cluster_labels에 저장. y_km의 고유값 개수는 클러스터의 개수와 동일함.

def plotSilhouette(x, pred):
    cluster_labels = np.unique(pred)
    n_clusters = cluster_labels.shape[0]   # 클러스터 개수를 n_clusters에 저장
    sil_val = silhouette_samples(x, pred, metric='euclidean')  # 실루엣 계수를 계산
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        # 각 클러스터에 속하는 데이터들에 대한 실루엣 값을 수평 막대 그래프로 그려주기
        c_sil_value = sil_val[pred == c]
        c_sil_value.sort()
        y_ax_upper += len(c_sil_value)

        plt.barh(range(y_ax_lower, y_ax_upper), c_sil_value, height=1.0, edgecolor='none')
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_sil_value)

    sil_avg = np.mean(sil_val)         # 평균 저장

    plt.axvline(sil_avg, color='red', linestyle='--')  # 계산된 실루엣 계수의 평균값을 빨간 점선으로 표시
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('클러스터')
    plt.xlabel('실루엣 개수')
    plt.show() 
'''    
    그래프를 보면 클러스터 1~3 에 속하는 데이터들의 실루엣 계수가 0으로 된 값이 아무것도 없으며, 실루엣 계수의 평균이 0.7 보다 크므로 잘 분류된 결과라 볼 수 있다.
'''
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
km = KMeans(n_clusters=3, random_state=0) 
y_km = km.fit_predict(X)

plotSilhouette(X, y_km)

















