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

def say_hello():
    return print("hello!!")

def Intersection(x,y):
    dim = x.shape[0]
    sim_val = 0# 교집합의 개수(둘다 1인 경우)
    for i in range(0,dim):
        if (x[i] == 1) and (y[i] == 1):
            sim_val += 1    

    return sim_val;

def jac_sim(x,y):
    dim = x.shape[0]
    (M01,M10,M00,M11) = 0,0,0,0
    for i in range(0,dim):
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
    dim = x.shape[0]
    (M01,M10,M00,M11) = 0,0,0,0
    for i in range(0,dim):
        if (x[i] == 0) and (y[i] == 1):
            M01 += 1
        elif (x[i] == 1) and (y[i] == 0):
            M10 += 1
        elif (x[i] == 0) and (y[i] == 0):
            M00 += 1
        elif (x[i] == 1) and (y[i] == 1):
            M11 += 1
    sim_val = (M00 + M11) / (M00 + M01 + M10 + M11)

    return sim_val;

def cos_sim(x,y):
    dim = x.shape[0]
    x = x.astype("float")
    y = y.astype("float")
    len_x = 0 # x의 크기(||x||)
    len_y = 0 # y의 크기(||y||)
    dot_x_y = 0 # x와 y의 내적(x·y)
    
    for i in range(0,dim):
        len_x += x[i]**2
        len_y += y[i]**2
        dot_x_y = x[i] * y[i]
    
    len_x = np.sqrt(len_x) # square root
    len_y = np.sqrt(len_y) # square root
    
    sim_val = dot_x_y / (len_x * len_y)

    return sim_val;

def man_dist(x,y):
    dim = x.shape[0]
    dist_val = 0 # dist의 최소값은 "0", 최대값은 무한대
    for i in range(0,dim):
        dist_val += np.abs(x[i] - y[i])
    return dist_val;


def euc_distance(x,y):
    dim = x.shape[0]
    dist_val = 0
    for i in range(dim):
        dist_val = dist_val + np.abs(x[i]-y[i])**2
    
    dist_val = np.sqrt(dist_val)# square root
    return dist_val;

def sup_dist(x,y):
    return np.abs(x-y).max();

def ham_dist(x,y):
    dim = x.shape[0]
    dist_val = 0
    for i in range(dim):
        if x[i] != y[i]:
            dist_val += 1
    return dist_val;

def corr(x,y):
    dim = x.shape[0]
    numerator = 0
    denominator_x = 0
    denominator_y = 0
    for i in range(dim):
        numerator += (x[i]-x_bar)*(y[i]-y_bar)
        denominator_x += (x[i]-x_bar)**2
        denominator_y += (y[i]-y_bar)**2
    denominator = np.sqrt(denominator_x + denominator_y)
    corr_val = numerator / denominator

    return corr_val;

# =========================================
# SSE
# =========================================
def Calc_SSE(clusters, X):

    sse = 0

    for j in np.unique(clusters):

        points = X[clusters == j]

        if len(points) == 0:
            continue

        center = points.mean(axis=0)

        sse += np.sum((points - center) ** 2)

    return sse


# =========================================
# Purity
# =========================================
def Calc_purity(clusters, Y):

    purity_sum = 0

    for i in np.unique(clusters):

        arr = Y[clusters == i]

        values, counts = np.unique(
            arr,
            return_counts=True
        )

        if len(counts) > 0:
            purity_sum += np.max(counts)

    result = purity_sum / len(Y)

    return result


# =========================================
# K-means
# =========================================
def partitional_clustering(X, k):

    num, dim = X.shape

    max_iter = 100

    # 서로 다른 k개의 초기 centroid 선택
    idx = np.random.choice(
        num,
        k,
        replace=False
    )

    old_m = X[idx, :].copy()

    for iters in range(max_iter):

        # ---------------------------------
        # 가장 가까운 centroid에 할당
        # ---------------------------------
        assign_mat = np.zeros(
            num,
            dtype=int
        )

        for i in range(num):

            distances = np.linalg.norm(
                X[i, :] - old_m,
                axis=1
            )

            assign_mat[i] = np.argmin(
                distances
            )

        # ---------------------------------
        # 새로운 centroid 계산
        # ---------------------------------
        new_m = np.zeros([k, dim])

        for j in range(k):

            points = X[assign_mat == j, :]

            if len(points) > 0:

                new_m[j, :] = points.mean(
                    axis=0
                )

            else:

                # 빈 cluster이면 기존 centroid 유지
                new_m[j, :] = old_m[j, :]

        # ---------------------------------
        # 수렴 확인
        # ---------------------------------
        if np.linalg.norm(old_m.reshape(1, k*dim)-new_m.reshape(1, k*dim)) < 1e-6:
            break

        old_m = new_m

    return assign_mat