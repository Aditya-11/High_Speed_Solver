import numpy as num
from random import randrange
from scipy.sparse.linalg import gmres 
import matplotlib.pyplot as plt
import math
import datetime

# gmmres with givens rotation
#givens rotation 

def gmres_algorithm_givens (A , b , x0 , error , max_iter ):
    
    res = b - num.asarray(num.dot(A,x0)).reshape(-1) # residual error

    x_pred = []

    q_ = [0] * max_iter

    x_pred.append(res)

    q_[0] = res / num.linalg.norm(res)

    h_ = num.zeros((max_iter + 1, max_iter))

    sn = num.zeros((max_iter , 1))

    cs = num.zeros((max_iter , 1))

    b_ = num.zeros(max_iter + 1)

    b_[0] = num.linalg.norm(res)

    global cpu_mul 

    cpu_mul = 0
    
    for k in range(min(max_iter , A.shape[0])) : 

        time_s1 = datetime.datetime.now()

        y_out = num.asarray(num.dot(A,q_[k])).reshape(-1)

        time_end1 = datetime.datetime.now()

        cpu_mul += (time_end1 - time_s1).total_seconds()

        for j in range(k+1) : 
            h_[j , k] = num.dot(q_[j],y_out)
            y_out = y_out - h_[j , k] * q_[j]
        
        h_[k+1 , k] = num.linalg.norm(y_out) 

        if (h_[k + 1, k] != 0 and k != max_iter - 1):
            q_[k+1] = y_out / h_[k+1 , k]
        
        for i in range(k): 
            temp   =  cs[i] * h_[i , k] + sn[i] * h_[i+1  , k]
            h_[i+1 ,k] = -1*sn[i] * h_[i , k] + cs[i] * h_[i+1 , k]
            h_[i , k]   = temp
        
        t = math.sqrt(h_[k , k]**2 + h_[k+1 , k]**2)

        cs[k] = (h_[k , k]) /t
        sn[k] = (h_[k+1 , k]) /t

        h_[k , k] = cs[k] * h_[k , k] + sn[k] * h_[k + 1 , k]
        h_[k + 1 , k] = 0

        b_[k + 1] = -1 * sn[k] * b_[k]
        b_[k] = cs[k] * b_[k]

        #print(h_)
        #print(b_)

        c_ = num.linalg.lstsq(h_ , b_)[0] 

        prod_ = num.asarray(num.dot(num.asarray(q_).transpose() , c_))

        if (k == max_iter - 1) :
            print('q_ ' + str(num.asarray(q_).shape) + ' c_shape = ' + str(c_.shape) + ' prod_ = ' + str(prod_.shape))

        #print(prod_)

        x_pred.append(prod_ + x0)  

        x_temp_ = (num.linalg.norm(b - num.dot(A ,(prod_ + x0)).reshape(-1)) / num.linalg.norm(b))

        g_1.append(math.log10(x_temp_))

        print(x_temp_)

        if (x_temp_ < error) :
            print("exit val : ", k)
            break

    return x_pred
