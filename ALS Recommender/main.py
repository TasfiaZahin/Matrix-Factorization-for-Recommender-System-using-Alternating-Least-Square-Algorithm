import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv
import math
import operator
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer
import sklearn as sk
import datetime
import random
import sys
import time
import pickle

tolerance = 0.01
max_itn = 100

empty_val = 88.0
empty_test = 77.0

class Index:
    def __init__(self,i,j):
        self.i = i
        self.j = j


def initialize(K, N, M):

    u = np.random.rand(N,K)
    v = np.random.rand(K,M)
    #print("u\n",u)
    #print("v\n",v)

    return u,v

def getRMSEtest(uv, dataset, org_dataset):

    # print("in RMSE")
    # print(uv)
    # print(dataset)
    #print(org_dataset)

    #calc error on places where val_empty = 88

    sum = 0.0
    count = 0
    ans = -1
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if(dataset[i][j] == empty_test):
                sum += (org_dataset[i][j] - uv[i][j])**2
                count += 1

    if(count != 0):
        ans = np.sqrt(float(sum/count))
    else:
        print("divide by zero")
    #print(ans)
    return ans

def getRMSEtrain(uv, dataset, org_dataset):

    # print("in RMSE")
    # print(uv)
    # print(dataset)
    #print(org_dataset)

    #calc error on places where val_empty = 88

    sum = 0.0
    count = 0
    ans = -1
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if(dataset[i][j] != 99 and dataset[i][j] != 88 and dataset[i][j] != 77):
                sum += (org_dataset[i][j] - uv[i][j])**2
                count += 1
    if (count != 0):
        ans = np.sqrt(float(sum / count))
    else:
        print("divide by zero")
        # print(ans)
    return ans

def getRMSEval(uv, dataset, org_dataset):

    # print("in RMSE")
    # print(uv)
    # print(dataset)
    #print(org_dataset)

    #calc error on places where val_empty = 88

    sum = 0.0
    count = 0
    ans = -1
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if(dataset[i][j] == empty_val):
                sum += (org_dataset[i][j] - uv[i][j])**2
                count += 1
    if (count != 0):
        ans = np.sqrt(float(sum / count))
    else:
        print("divide by zero")
        # print(ans)
    return ans

def getIndices(dataset):

    idxs = []
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if(dataset[i][j] == empty_val):
                idx = Index(i,j)
                idxs.append(idx)

    return idxs

def getErr(uv, org_dataset, idxs):
    sum = 0.0
    count = 0
    ans = -1
    for i in range(len(idxs)):
        idx = idxs[i]
        sum += (org_dataset[idx.i][idx.j] - uv[idx.i][idx.j]) ** 2
        count += 1

    if (count != 0):
        ans = np.sqrt(float(sum / count))
    else:
        print("divide by zero")
        # print(ans)
    return ans


def runALS(dataset, K, lamb_u, lamb_v, org_dataset):

    N = len(dataset)
    M = len(dataset[0])
    u,v = initialize(K, N, M)

    uv = u.dot(v)
    old_err = getRMSEtrain(uv, dataset, org_dataset)
    #print("initial train err:",old_err)
    #print("uv\n", uv)

    #idxs_err = getIndices(dataset)
    c = 1
    while 1:

        #print("Iteration:", c)
        #print("u:\n",u)
        #print("v:\n",v)

        #finding v
        for m in range(M):
            mth_column = dataset[:,m]
            #print(mth_column)

            un_unT = np.zeros((K,K))
            sec_part = np.zeros((K, 1))

            # #print("dattt\n",dataset)
            # idxs = np.argwhere((mth_column != 99) & (mth_column != 88) & (mth_column != 77))
            # #print("idxs\n",idxs)
            #
            # for d in range (len(idxs)):
            #     n = int(idxs[d])
            #     #print("n",n)
            #     un = np.reshape(u[n], (len(u[n]), 1))
            #     un_unT += un.dot(np.transpose(un))
            #     sec_part += np.multiply(mth_column[n], un)

            #x = 0
            for n in range (len(mth_column)):
                if(mth_column[n] != 99 and mth_column[n] != 88 and mth_column[n] != 77):
                    un = np.reshape(u[n],(len(u[n]),1))
                    #print(un)
                    #print("in looppppppp")
                    un_unT += un.dot(np.transpose(un))
                    sec_part += np.multiply(mth_column[n], un)
                #x += 1
            #print(np.multiply(lamb_v, np.identity(K)))
            un_unT += np.multiply(lamb_v, np.identity(K))
            #print(un_unT)
            #print("x",x)

            inv = np.linalg.inv(un_unT)
            vm = inv.dot(sec_part)

            #print(vm)
            v[:,m] = np.reshape(vm,(1,len(vm)))
            #print(v[:,m])
        #print("updated v\n",v)

        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        #finding u
        for n in range(N):
            nth_row = dataset[n]
            #print(nth_row)

            vm_vmT = np.zeros((K,K))
            sec_part = np.zeros((K, 1))

            # #print("dattt\n",dataset)
            # idxs = np.argwhere((nth_row != 99) & (nth_row != 88) & (nth_row != 77))
            # #print("idxs\n",idxs)
            #
            # for d in range(len(idxs)):
            #     m = int(idxs[d])
            #     #print("m", m)
            #     vm = np.reshape(v[:, m], (len(v[:, m]), 1))
            #     vm_vmT += vm.dot(np.transpose(vm))
            #     sec_part += np.multiply(nth_row[m], vm)

            for m in range(len(nth_row)):
                if (nth_row[m] != 99 and nth_row[m] != 88 and nth_row[m] != 77):
                    vm = np.reshape(v[:,m],(len(v[:,m]),1))
                    #print(vm)
                    vm_vmT += vm.dot(np.transpose(vm))
                    sec_part += np.multiply(nth_row[m], vm)
            #print(np.multiply(lamb_u, np.identity(K)))
            vm_vmT += np.multiply(lamb_u, np.identity(K))
            #print(vm_vmT)

            inv = np.linalg.inv(vm_vmT)
            un = inv.dot(sec_part)

            #print(un)
            u[n] = np.reshape(un,(1,len(un)))
            #print(u[n])
        #print("updated u\n",u)

        uv = u.dot(v)
        #print("uv\n",uv)

        #calculate RMSE

        #new_err = getErr(uv,org_dataset,idxs_err)
        new_err = getRMSEtrain(uv, dataset, org_dataset)
        #print("K=",K," itn:",c," train err:", new_err)
        diff = new_err - old_err
        #print("diff:", diff)

        #if (diff >= tolerance and diff < 0):
        if (abs(diff) <= tolerance):
            break

        old_err = new_err

        c += 1
        if (c > max_itn):
            break
    print("no of itns:",c-1)
    print("train error for current hypers:", new_err)
    return uv

def getBestHyperparameters(test_empty, org_dataset, Ks, lambs):

    best_err = math.inf
    best_K = None
    best_lamb_u = None
    best_lamb_v = None

    for i in range (len(Ks)):
        for j in range (len(lambs)):
            #for k in range(len(lambs)):
                K = Ks[i]
                lamb_u = lambs[j]
                lamb_v = lambs[j]
                # lamb_u = 0.01
                # lamb_v = 0.01
                print("Running ALS for K =",K," lamb_u =",lamb_u," lamb_v =",lamb_v)
                uv = runALS(test_empty, K, lamb_u, lamb_v, org_dataset)
                err = getRMSEval(uv, test_empty, org_dataset)
                print("val error for current hypers:",err)
                if(err < best_err):
                    best_err = err
                    best_K = K
                    best_lamb_u = lamb_u
                    best_lamb_v = lamb_v


    print("Best values K =", best_K, " lamb_u =", best_lamb_u, " lamb_v =", best_lamb_v)
    return best_K, best_lamb_u, best_lamb_v


def runALStrain(dataset, K, lamb_u, lamb_v, org_dataset):

    N = len(dataset)
    M = len(dataset[0])
    u,v = initialize(K, N, M)

    uv = u.dot(v)
    old_err = getRMSEtest(uv, dataset, org_dataset)
    #print("initial train err:", old_err)

    c = 1
    while 1:

        #print("Iteration:", c)

        #finding v
        for m in range(M):
            mth_column = dataset[:,m]
            #print(mth_column)

            un_unT = np.zeros((K,K))
            sec_part = np.zeros((K, 1))

            # # print("dattt\n",dataset)
            # idxs = np.argwhere((mth_column != 99) & (mth_column != 88) & (mth_column != 77))
            # # print("idxs\n",idxs)
            #
            # for d in range (len(idxs)):
            #     n = int(idxs[d])
            #     #print("n",n)
            #     un = np.reshape(u[n], (len(u[n]), 1))
            #     un_unT += un.dot(np.transpose(un))
            #     sec_part += np.multiply(mth_column[n], un)

            for n in range (len(mth_column)):
                if(mth_column[n] != 99 and mth_column[n] != 77):
                    un = np.reshape(u[n],(len(u[n]),1))
                    #print(un)
                    un_unT += un.dot(np.transpose(un))
                    sec_part += np.multiply(mth_column[n], un)
            #print(np.multiply(lamb_v, np.identity(K)))
            un_unT += np.multiply(lamb_v, np.identity(K))
            #print(un_unT)

            inv = np.linalg.inv(un_unT)
            vm = inv.dot(sec_part)

            #print(vm)
            v[:,m] = np.reshape(vm,(1,len(vm)))
            #print(v[:,m])
        #print("updated v\n",v)

        #finding u
        for n in range(N):
            nth_row = dataset[n]
            #print(nth_row)

            vm_vmT = np.zeros((K,K))
            sec_part = np.zeros((K, 1))

            # # print("dattt\n",dataset)
            # idxs = np.argwhere((nth_row != 99) & (nth_row != 88) & (nth_row != 77))
            # # print("idxs\n",idxs)
            #
            # for d in range(len(idxs)):
            #     m = int(idxs[d])
            #     # print("m", m)
            #     vm = np.reshape(v[:, m], (len(v[:, m]), 1))
            #     vm_vmT += vm.dot(np.transpose(vm))
            #     sec_part += np.multiply(nth_row[m], vm)

            for m in range(len(nth_row)):
                if (nth_row[m] != 99 and nth_row[m] != 77):
                    vm = np.reshape(v[:,m],(len(v[:,m]),1))
                    #print(vm)
                    vm_vmT += vm.dot(np.transpose(vm))
                    sec_part += np.multiply(nth_row[m], vm)
            #print(np.multiply(lamb_u, np.identity(K)))
            vm_vmT += np.multiply(lamb_u, np.identity(K))
            #print(vm_vmT)

            inv = np.linalg.inv(vm_vmT)
            un = inv.dot(sec_part)

            #print(un)
            u[n] = np.reshape(un,(1,len(un)))
            #print(u[n])
        #print("updated u\n",u)

        uv = u.dot(v)
        #print("uv\n",uv)

        #calculate RMSE

        new_err = getRMSEtrain(uv, dataset, org_dataset)
        #print("K=", K, " itn:", c, " train err:", new_err)
        diff = new_err - old_err
        #print("diff:", diff)

        #if (diff >= tolerance and diff < 0):
        if(abs(diff) <= tolerance):
            break

        old_err = new_err

        c += 1
        if(c > max_itn):
            break

    print("no of itns:", c - 1)
    print("train error for current hypers:", new_err)
    return uv, new_err

def getValset(dataset, percent):

    val_set = dataset.copy()
    #print("train set\n",train_set)

    for i in range (len(dataset)):
        p = np.around(percent/100.0*dataset[i][0])
        #print(p)
        row = val_set[i]
        lst = []
        for j in range (1,len(val_set[0])):
            if (val_set[i][j] != 99.0):
                #print(j)
                lst.append(j)
        #print(lst)
        if (len(lst) != 0):
            rand_idx = np.random.choice(lst, int(p), replace=False)
            #print(rand_idx)
            for j in range(len(rand_idx)):
                row[int(rand_idx[j])] = empty_val
    #val_set = np.delete(val_set, 0, axis=1)
    #print("valset\n",val_set)
    return val_set

def getTestset(dataset, percent):

    test_set = dataset.copy()
    #print("train set\n",train_set)

    for i in range (len(dataset)):
        p = np.around(percent/100.0*dataset[i][0])
        #print(p)
        row = test_set[i]
        lst = []
        for j in range (1,len(test_set[0])):
            if (test_set[i][j] != 99.0 and test_set[i][j] != 88.0):
                #print(j)
                lst.append(j)
        #print(lst)
        if (len(lst) != 0):
            rand_idx = np.random.choice(lst, int(p), replace=False)
            #print(rand_idx)
            for j in range(len(rand_idx)):
                row[int(rand_idx[j])] = empty_test
    #test_set = np.delete(val_set, 0, axis=1)
    #print("testset\n",test_set)
    return test_set


def main():

    dataset = pd.read_csv("data.csv", delimiter=',', header=None)
    dataset = dataset.values
    org_dataset = dataset.copy()
    #print("original dataset\n",org_dataset)

    # random.seed(1)
    # np.random.seed(1)

    val_empty_set = getValset(dataset, 20.0)
    test_empty_set = getTestset(val_empty_set, 20.0)

    val_empty_set = np.delete(val_empty_set, 0, axis=1)
    test_empty_set = np.delete(test_empty_set, 0, axis=1)
    org_dataset = np.delete(org_dataset, 0, axis=1)

    print("to be sent\n",test_empty_set)
    #print(len(test_empty_set[0]))

    start = time.time()

    Ks = [5,10,20,40]
    lambs = [0.01,0.1,1,10]
    best_K, best_lamb_u, best_lamb_v = getBestHyperparameters(test_empty_set, org_dataset, Ks, lambs)
    #runALS(test_empty_set, 5, 0.01, 0.01, org_dataset)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    #print(end - start)

    #print(test_empty_set)
    #make empty_val = 88 values to prev values
    for i in range(len(test_empty_set)):
        for j in range (len(test_empty_set[0])):
            if(test_empty_set[i][j] == empty_val):
                #print("set new")
                test_empty_set[i][j] = org_dataset[i][j]

    #print(test_empty_set)

    #train again with best values and 80% data
    print("\nTraining on best hyperparameters...\n")
    uv, err = runALStrain(test_empty_set, best_K, best_lamb_u, best_lamb_v, org_dataset)

    print("final uv",uv)
    print("train err for 80% trainset",err)

    np.save("uv",uv)

    uv = np.load("uv.npy")

    # f = open('test', 'w')
    # pickle.dump(uv, f)
    # f.close()
    #
    # f2 = open('test', 'r')
    # uv = pickle.load(f2)
    # f2.close()

    # uv.dump("my_matrix.dat")
    #
    # uv = np.load("my_matrix.dat")

    RMSE_on_test = getRMSEtest(uv, test_empty_set, org_dataset)
    print("RMSE on test",RMSE_on_test)


main()

