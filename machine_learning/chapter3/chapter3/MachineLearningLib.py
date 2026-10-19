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
# import cv2

# 한글 형태소 분석
# from konlpy.tag import Okt, Kkma

# 머신러닝 - 데이터 분할 및 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

# 머신러닝 - 텍스트 벡터화 및 유사도
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 머신러닝 - 회귀
from sklearn.linear_model import LinearRegression

def say_hello():
    return print("hello!!")

def Intersection(x,y):
    dim = x.shape[0]
    sim_val = 0 # 교집합의 개수 (둘다 1인 경우)
    for i in range(0, dim):
        if (x[i] == 1) and (y[i] == 1):
            sim_val += 1
    

    return sim_val;

def jac_sim(x,y):
   (M01,M10,M00,M11) = 0,0,0,0
   dim = x.shape[0]
   for i in range(0, dim):
      if (x[i] == 0) and (y[i] == 1):
         M01 += 1
      elif (x[i] == 1) and (y[i] == 0):
         M10 += 1
      elif (x[i] == 0) and (y[i] == 0):
         M00 += 1
      elif (x[i] == 1) and (y[i] == 1):
         M11 += 1
      sim_val = (M11) / (M01 + M10 + M11)


   return sim_val;

def smc_sim(x,y):
    (M01,M10,M00,M11) = 0,0,0,0
    dim = x.shape[0]
    for i in range(0, dim):
        if (x[i] == 0) and (y[i] == 1):
            M01 += 1
        elif (x[i] == 1) and (y[i] == 0):
            M10 += 1
        elif (x[i] == 0) and (y[i] == 0):
            M00 += 1
        elif (x[i] == 1) and (y[i] == 1):
            M11 += 1
        sim_val = (M01 + M00) / (M01 + M10 + M11 + M00)

    return sim_val;

def cos_sim(x,y):
    dim = x.shape[0]
    len_x = 0 # x의 크기(||x||)
    len_y = 0 # y의 크기(||y||)
    dot_x_y = 0 # x와 y의 내적(x·y)

    for i in range(0,dim):
        len_x += x[i]**2
        len_y += y[i]**2
        dot_x_y += x[i] * y[i]

    len_x = np.sqrt(len_x) # square root
    len_y = np.sqrt(len_y) # square root
    sim_val = dot_x_y / (len_x * len_y)

    # print(len_x,len_y,dot_x_y)
    return sim_val;

def man_dist(x,y):
    dim = x.shape[0]
    dist_val = 0 # dist의 최소값은 0, 최대값은 무한대
    for i in range(0,dim):
        dist_val += np.abs(x[i]-y[i])


    return dist_val;