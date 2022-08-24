# 분류분석 - Decision Tree -
# - Decision Tree는 여러 가지 규칙을 순차적으로 적용하면서 독립 변수 공간을 분할하는 분류 모형이다. 
# 분류(classification)와 회귀 분석(regression)에 모두 사용될 수 있다. 해석이 쉽다.
# - 비모수 검정 : 선형성, 정규성, 등분산성 가정 필요 없음
# - 단점 : 유의수준 판단 기준 없음(추론 기능 없음), 비연속성/ 선형성 또는 주효과 결여/ 비안정성(분석용
# 자료에만 의존하므로)으로 새로운 자료의 예측에서는 불안정할 수 있음.

import collections
from sklearn import tree

x = [[180, 15],[177, 42],[156, 35],[174, 5],[166, 33],
     [180, 75],[167, 2],[166, 35],[174, 25],[168, 23]]
y = ['man', 'woman','woman','man','woman',
     'man','man','man','man','woman']
label_names =['height', 'hair length']

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
print(model)
fit = model.fit(x, y)
print('훈련 정확도:', model.score(x, y))

pred = model.predict(x)
print('예측값:', pred)
print('실제값:', y)

# 시각화
# pip install pydotplus
# pip install graphviz
import pydotplus
dot_data = tree.export_graphviz(model, feature_names = label_names,
                                out_file=None, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('red', 'orange')
edges = collections.defaultdict(list)  # list type 변수 준비

for e in graph.get_edge_list():
    edges[e.get_source()].append(int(e.get_destination()))
    
for e in edges: 
    edges[e].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[e][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')

# 엔트로피 = 혼잡도 <---> 균일도

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
img = imread('tree.png')
plt.imshow(img)
plt.show()

new_pred = model.predict([[170,120 ]])
print('새 예측값:', new_pred)

