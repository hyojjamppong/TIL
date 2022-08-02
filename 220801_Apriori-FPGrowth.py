import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


data = [
        ['우유', '빵', '과일'],
        ['우유', '버터'],
        ['버터', '달걀'],
        ['우유', '빵', '버터'],
        ['우유', '빵', '기저귀', '버터', '과일'],
        ['빵', '기저귀', '과일'],
        ['빵', '과일', '기저귀'],
        ]


#  데이터 형식 변경 및 학습
te = TransactionEncoder()
te_result = te.fit(data).transform(data)
te_result


df = pd.DataFrame(te_result, columns=te.columns_)
df = df.astype(int)


# 기존의 리스트 형태로 돌리려면?
# te.inverse_transform(te_result)


# 인덱스 대신 DataFrame의 열 이름 반환
item = apriori(df, use_colnames=True, min_support=0.4)
item

# confidenc >= 0.6
association_rules(item, metric='confidence',  min_threshold=0.6)
