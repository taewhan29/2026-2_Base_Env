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

def partitional_clustering(X,k): # X는 데이터셋, k는 클러스터 갯수
    MAX_ITER = 100 # 최대 반복횟수
    num, dim = X.shape
    # k개의 랜덤 초기값 M
    M = X[np.random.randint(0,num,[k]),:]

    # k-means clustering 수행
    for iters in range(MAX_ITER): # 최대 반복 횟수
        assign_mat = np.zeros(num)
        for i in range(0,num):
            dist = np.zeros(k)
            for j in range(k):
                dist[j] = euc_distance(X[i,:],M[j,:])
            assign_mat[i] = np.argmin(dist)
            
        new_M = np.zeros((k,dim))
        for j in range(k):
            new_M[j,:] = X[assign_mat == j,:].mean(axis=0)
            
        # print(iters, M, new_M)
        if (euc_distance(M.flatten(),new_M.flatten()) == 0):
            break
        else:
            M = new_M  

    assign_mat = np.zeros(num)
    for i in range(0,num):
        dist = np.zeros(k)
        for j in range(k):
            dist[j] = euc_distance(X[i,:],M[j,:])
        assign_mat[i] = np.argmin(dist)
    return assign_mat


def Calc_purity(clusters,Y):
    purity = 0
    for g in np.unique(clusters):
        group = Y[clusters==g]
        val, cnt = np.unique(group, return_counts=True)
        if len(val) != 0:            
            purity += np.max(cnt)/np.sum(cnt)
    
    return purity

def Calc_SSE(clusters,X):
    sse = 0
    for c in np.unique(clusters):
        # 그룹 나누기
        group = X[clusters == c,:]
        if group.shape[0] > 0:
            # 평균 구하기
            m = np.mean(group, axis = 0)
            # 거리(에러) 구하기
            for i in range(len(group)):
                sse += euc_distance(group[i,:], m)
    return sse