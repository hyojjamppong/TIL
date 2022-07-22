import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("https://bit.ly/seoul-120-text-csv")
df["문서"] = df["제목"] + " " + df["내용"]  # 문서라는 파생변수 만들기

df["분류"].value_counts(normalize=True) * 100 # 분류별 빈도수 확인
df.loc[~df['분류'].isin(['행정','경제'], "분류") = "기타" # 일부 상위 분류 데이터 추출
df["분류"].value_counts(normalize=True) * 100

       
label_name = "분류"
X = df["문서"]
y = df["분류"]
y_ohe = pd.get_dummies(y)

       
from sklearn.model_selection import train_test_split
       
X_train_text_all, X_test_text, y_train_all, y_test = train_test_split(
  X, y_ohe, test_size=0.2, random_state=42, stratify=y_ohe)
       
X_train_text, X_valid_text, y_train, y_valid = train_test_split(
  X_train_text_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all)

       
from tensorflow.keras.preprocessing.text import Tokenizer
vocab_size = 40063 # 텍스트 데이터의 전체 단어 집합의 크기
oov_tok = "<oov>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token = oov_tok)
tokenizer
tokenizer.fit_on_texts(X_train_text)
       
       
len(tokenizer.word_index) # 각 인덱스에 해당하는 단어 확인
list(tokenizer.word_counts.items())[:5] # 단어별 빈도수 확인
pd.DataFrame(tokenizer.word_counts.items()).set_index(0).nlargest(10, 1).T  # 단어별 빈도수 확인     

       
# 텍스트 문장을 숫자로 이루어진 리스트로 변경       
train_sequences = tokenizer.texts_to_sequences(X_train_text)
valid_sequences = tokenizer.texts_to_sequences(X_valid_text)
test_sequences = tokenizer.texts_to_sequences(X_test_text)

       
# Padding
df["문서"].map(lambda x : len(x.split())).describe()
max_length = 100 패딩의 기준
padding_type = "post"
X_train = pad_sequences(train_sequences, padding='post', maxlen=max_length)
X_valid = pad_sequences(valid_sequences, padding='post', maxlen=max_length)
X_test = pad_sequences(test_sequences, padding='post', maxlen=max_length)
       
# Modeling
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, GRU, Bidirectional, LSTM, Dropout, BatchNormalization      

embedding_dim = 64 # 임베딩 할 벡터의 차원
n_class = y_train.shape[1] # 분류될 예측값의 종류 
       
       
model = Sequential()
model.add(Embedding(input_dim=vocab_size, 
                    output_dim=embedding_dim, 
                    input_length=max_length))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(LSTM(units=64))
model.add(Dense(n_class, activation="softmax"))       
model.summary()

       
# Model Compile
model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])
       
# Learning
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)
       
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10)       
       

# Result       
df_hist = pd.DataFrame(history.history)
df_hist.head()
       
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
df_hist[["loss", "val_loss"]].plot(ax=axes[0])
df_hist[["accuracy", "val_accuracy"]].plot(ax=axes[1])
       
# Predict      
y_pred = model.predict(X_test)
       
       
# Evaluate      
y_predict = np.argmax(y_pred, axis=1)       
(y_predict == y_test_val).mean()       
       
       
       
       
       
