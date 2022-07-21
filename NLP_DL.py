import pandas as pd
corpus = ["서울 코로나 상생지원금 문의입니다.",
"인천 지하철 운행시간 문의입니다.",
"버스 운행시간 문의입니다."]

from tensorflow.keras.preprocessing.text import Tokenizer
vocab_size = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<oov>")
tokenizer.fit_on_texts(corpus)

tokenizer.word_index  # 인덱스 해당 단어 확인
tokenizer.word_counts.items() # 단어별 빈도수 확인
pd.DataFrame(tokenizer.word_counts.items()).set_index(0).sort_values(by=1).T
corpus_sentences = tokenizer.text_to_sequences(corpus)

from tensorflow.keras.preprocessing.sequence import pad_sequence
pad_sequence(corpus_sentences, max_len=10, padding="pre", truncating="pre")
