# 상관 관계 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

data = pd.read_csv('drinking_water.csv')
print(data.head(3), len(data))

print('공분산')
print(np.cov(data.친밀도, data.적절성))
print(np.cov(data.친밀도, data.만족도))
print()
print(data.cov())

print()
print('상관계수')
print(np.corrcoef(data.친밀도, data.적절성))
print(np.corrcoef(data.친밀도, data.만족도))
print(data.corr(method='pearson'))  # 등간, 비율 척도인 경우, 정규성을 따름
# print(data.corr(method='spearman')) # 서열척도인 경우 사용
# print(data.corr(method='kendall'))  # spearman과 유사

print()
# 시각화 
import seaborn as sns
sns.heatmap(data.corr())
plt.show()

# 다른 시각화 방법
# heatmap에 텍스트 표시 추가사항 적용해 보기
corr = data.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)  # 상관계수값 표시
mask[np.triu_indices_from(mask)] = True
# Draw the heatmap with the mask and correct aspect ratio
vmax = np.abs(corr.values[~mask]).max()
fig, ax = plt.subplots()     # Set up the matplotlib figure

sns.heatmap(corr, mask=mask, vmin=-vmax, vmax=vmax, square=True, linecolor="lightgray", linewidths=1, ax=ax)

for i in range(len(corr)):
    ax.text(i + 0.5, len(corr) - (i + 0.5), corr.columns[i], ha="center", va="center", rotation=45)
    for j in range(i + 1, len(corr)):
        s = "{:.3f}".format(corr.values[i, j])
        ax.text(j + 0.5, len(corr) - (i + 0.5), s, ha="center", va="center")
ax.axis("off")
plt.show()