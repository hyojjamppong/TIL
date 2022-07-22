import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://bit.ly/seoul-120-text-csv")
df["문서"] = df["제목"] + " " + df["내용"]  # 문서라는 파생변수 만들기

df["분류"].value_counts(normalize=True) * 100 # 분류별 빈도수 확인
df.loc[~df['분류'].isin(['행정','경제'], "분류") = "기타" # 일부 상위 분류 데이터 추출
df["분류"].value_counts(normalize=True) * 100
