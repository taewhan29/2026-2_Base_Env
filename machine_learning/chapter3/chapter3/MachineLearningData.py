# 기본 라이브러리
import re
import tarfile
import urllib.request
from pathlib import Path

# 데이터 처리 및 시각화
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# 한글 형태소 분석
from konlpy.tag import Okt, Kkma

# 머신러닝 - 데이터 분할 및 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

# 머신러닝 - 텍스트 벡터화 및 유사도
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 머신러닝 - 회귀
from sklearn.linear_model import LinearRegression

def gen_doc():
    # 문서 데이터
    docs = {
        1 : "먹고 싶은 사과",
        2 : "먹고 싶은 바나나",
        3 : "길고 노란 바나나 바나나",
        4 : "저는 과일이 좋아요"
    }
    
    # 1) 문서별 토큰화
    tokenized_docs = {}
    for doc_id, text in docs.items():   # 각 문서 ID와 텍스트 순회
        tokens = text.split()           # 공백 단위로 토큰화
        tokenized_docs[doc_id] = tokens # 결과를 딕셔너리에 저장
    
    # 2) 고유 단어 집합 생성
    unique_token = []
    for tokens in tokenized_docs.values():
        for token in tokens:
            unique_token.append(token)
    unique_token = set(unique_token)
    
    num = len(docs)
    dim = len(unique_token)
    
    data = np.zeros((num, dim))
    for i, (tid,tokens) in enumerate(tokenized_docs.items()):
        for j, token in enumerate(unique_token):
            if token in tokens:
                data[i,j] = 1
    
    return np.int32(data)