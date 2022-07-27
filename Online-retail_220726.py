import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt


# font setting
def get_font_family():
    """
    시스템 환경에 따른 기본 폰트명을 반환하는 함수
    """
    import platform
    system_name = platform.system()

    if system_name == "Darwin" :
        font_family = "AppleGothic"
    elif system_name == "Windows":
        font_family = "Malgun Gothic"
    else:
        # Linux(Colab)
        !apt-get install fonts-nanum -qq  > /dev/null
        !fc-cache -fv

        import matplotlib as mpl
        mpl.font_manager._rebuild()
        findfont = mpl.font_manager.fontManager.findfont
        mpl.font_manager.findfont = findfont
        mpl.backends.backend_agg.findfont = findfont
        
        font_family = "NanumBarunGothic"
    return font_family

plt.rc("font", family=get_font_family())
plt.rc("axes", unicode_minus=False)


# Data Load
df = pd.read_csv("data/commerce/online_retail.csv")


# Data preprocessing
df.head(2)
df.tail(2)
df.info()
df.describe()
df.describe(include='object')
df.isnull().sum()
df.isnull().mean() * 100


# 결측치 시각화
plt.figure(figsize=(12, 4))
sns.heatmap(df.isnull(), cmap="gray")


# 전체 수치 변수 시각화
_ = df.hist()


# 파생변수 생성
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"] # 전체 주문 금액


# 회원/비회원 구매 비교
df.loc[df['CustomerID'].isnull(), 'Country'].value_counts() # 국가별 비회원 수
df.loc[~df['CustomerID'].isnull(), 'Country'].value_counts() # 국가별 회원 수


# 매출액 상위 국가
df.groupby('Country')['TotalPrice'].agg(['mean', 'sum']).nlargest(10, 'sum').style.format("{:,.0f}")


# 상품 판매 빈도, 판매 총 수량, 총 매출액 상위 10개
stock_sale = df.groupby("StockCode").aggregate({"InvoiceNo" : "count", 
                                                "Quantity" : "sum",
                                                "TotalPrice" : "sum"
                                                 }).nlargest(10, "InvoiceNo")

stock_desc = df.loc[df["StockCode"].isin(stock_sale.index),
                    ["StockCode", "Description"]].drop_duplicates("StockCode").set_index("StockCode")
stock_desc.loc[stock_sale.index] # 해당 제품명 확인
stock_sale["Desc"] = stock_desc.loc[stock_sale.index] # 상품명 컬럼 생성
stock_sale


# 구매 취소
df["Cancel"] = df["Quantity"] < 0 
df.groupby("CustomerID")["Cancel"].value_counts(normalize=True).unstack()
df.groupby("CustomerID")["Cancel"].value_counts().unstack().nlargest(10, False)


# 특정 고객 구매 건 조회
df[df["CustomerID"] == 17841] # 취소 주문


# 제품별 취소 비율
cancel_stock = df.groupby("StockCode").agg({"InvoiceNo" : "count", "Cancel" : "mean"})
cancel_stock.nlargest(10, "InvoiceNo")


# 국가별 구매 취소 비율
cancel_country = df.groupby("Country").agg({"InvoiceNo" : "count", "Cancel" : "mean"})
cancel_country.nlargest(20, "Cancel")


# 날짜 변환 후 파생변수 생성
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["InvoiceYear"] = df["InvoiceDate"].dt.year
df["InvoiceMonth"] = df["InvoiceDate"].dt.month
df["InvoiceDay"] = df["InvoiceDate"].dt.day
df["InvoiceDow"] = df["InvoiceDate"].dt.dayofweek
df["InvoiceYM"] = df["InvoiceDate"].astype(str).str[:7] # 년월
df["InvoiceTime"] = df["InvoiceDate"].dt.time
df["InvoiceHour"] = df["InvoiceDate"].dt.hour


# 전체 변수 시각화
_ = df.hist(figsize=(12, 12), bins=50)
plt.figure(figsize=(12, 4))


# 구매 빈도수 시각화
plt.figure(figsize=(12, 4))
sns.countplot(data=df, x="InvoiceYear") # 연도별
sns.countplot(data=df, x="InvoiceMonth") # 월별
sns.countplot(data=df, x="InvoiceYM") # 연도월별
sns.countplot(data=df, x="InvoiceDow") # 요일별
sns.countplot(data=df, x="InvoiceDow", hue="Cancel") # 요일별 구매와 취소 빈도수
sns.countplot(data=df[df["Cancel"]], x="InvoiceDow") # 요일별 최소 빈도수
