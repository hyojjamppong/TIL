import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("http://bit.ly/data-diabetes-csv")
df.shape


# 학습, 예측 데이터셋 나누기
df.shape

# 8:2 의 비율로 구하기 위해 전체 데이터의 행에서 80% 위치에 해당되는 값을 구해서 split_count 라는 변수에 담기
split_count = int(df.shape[0] * 0.8)
split_count

# train, test로 슬라이싱을 통해 데이터를 나누기
train = df.iloc[:split_count]
train.shape
test = df.iloc[split_count:]
test.shape

train.shape[0] + test.shape[0] == df.shape[0]
True

# 정답값이자 예측해야 될 값
## label_name 이라는 변수에 예측할 컬럼의 이름 담기
label_name = "Outcome"
label_name

# 학습, 예측에 사용할 컬럼
## feature_names 라는 변수에 학습과 예측에 사용할 컬럼명을 가져오기
feature_names = df.columns.tolist()
feature_names.remove(label_name)
feature_names

# 학습, 예측 데이터셋 만들기
## 학습 세트 만들기 예) 시험의 기출문제
X_train = train[feature_names]
print(X_train.shape)
X_train.head()

# 정답 값을 만들어 줍니다. 예) 기출문제의 정답
y_train = train[label_name]
print(y_train.shape)

# 예측에 사용할 데이터세트를 만듭니다. 예) 실전 시험 문제
X_test = test[feature_names]
print(X_test.shape)
X_test.head()

# 예측의 정답값 예) 실전 시험 문제의 정답
y_test = test[label_name]
print(y_test.shape)
y_test.head()

# max_depth == 1 트리의 깊이를 의미한다
# max_teatures == 0.9 라면 전체 피처의 90%만 사용한다


# 머신러닝 알고리즘 가져오기
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=6, 
                               max_features=0.8,
                               max_leaf_nodes=25,
                               random_state=42)
model
y_train.head()

# 학습(훈련)
model.fit(X_train, y_train)

# 예측
y_predict = model.predict(X_test)
y_predict

# 트리 알고리즘 분석하기 (의사결정나무 시각화)
feature_names[1]
# plot_tree 를 통해 시각화 합니다.
from sklearn.tree import plot_tree

plt.figure(figsize=(20,20))
plot_tree(model, filled=True, fontsize=14, feature_names=feature_names)
plt.show()

# 피처의 중요도를 추출하기
np.sum(model.feature_importances_)
model.feature_importances_

# 피처의 중요도 시각화 하기
sns.barplot(x=model.feature_importances_, y=feature_names)

# 정확도(Accuracy) 측정하기
# 예측의 정확도를 구합니다. 100점 만점 중에 몇 점을 맞았는지 구한다고 보면 됩니다.
(y_test == y_predict).mean()

# 위에서 처럼 직접 구할 수도 있지만 미리 구현된 알고리즘을 가져와 사용합니다.
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)

# model 의 score 로 점수를 계산합니다.
model.score(X_test, y_test)
