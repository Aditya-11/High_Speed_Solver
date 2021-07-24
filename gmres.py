import numpy as num
from random import randrange
from scipy.sparse.linalg import gmres 
import matplotlib.pyplot as plt
import math
import datetime


def gen_matrix(n1) : 
    a1 = ''

    for i in range(n1): 

        for j in range(n1): 
            a1 += str(randrange(n1*10))
            a1 += ' '

        if i != n1-1:
            a1 += ';'
            a1 += ' '

    return num.matrix(a1)

def _plot_graph ():
    plt.plot(range(len(g_1)) , g_1 , color='black')
    plt.xlabel('N') 
    plt.ylabel('error')
    plt.title('plot')


def gmres_algorithm (A , b , x0 , error , max_iter ):
    
    res = b - num.asarray(num.dot(A,x0)).reshape(-1) # residual error

    #print ("res  " , res)

    x_pred = []

    q_ = [0] * max_iter

    x_pred.append(res)

    q_[0] = res / num.linalg.norm(res)

    #print("q_   " ,  q_)

    h_ = num.zeros((max_iter + 1, max_iter))

    for k in range(min(max_iter , A.shape[0])) : 
        y_out = num.asarray(num.dot(A,q_[k])).reshape(-1)

        #print (" y_out : " , y_out)

        for j in range(k+1) : 
            h_[j , k] = num.dot(q_[j],y_out)
            y_out = y_out - h_[j , k] * q_[j]

        #print ("y_out  :  " , y_out)
        
        h_[k+1 , k] = num.linalg.norm(y_out) 

        if (h_[k + 1, k] != 0 and k != max_iter - 1):
            q_[k+1] = y_out / h_[k+1 , k]

        b_ = num.zeros(max_iter + 1)
        b_[0] = num.linalg.norm(res)

        c_ = num.linalg.lstsq(h_ , b_)[0] 

        prod_ = num.asarray(num.dot(num.asarray(q_).transpose() , c_))

        if (k == max_iter - 1) :
            print('q_ ' + str(num.asarray(q_).shape) + ' c_shape = ' + str(c_.shape) + ' prod_ = ' + str(prod_.shape))

        x_pred.append(prod_ + x0)  

        #print ("h_  : " , h_)

        #print ("b_ : " , b_)

        #print ("x_pred  " , prod_ + x0 )

        x_temp_ = (num.linalg.norm(b - num.dot(A ,(prod_ + x0)).reshape(-1)) / num.linalg.norm(b))

        g_1.append(math.log10(x_temp_))

    return x_pred

if __name__ == '__main__' :

    lis_B1 = []

    lis_C1 = []

    number = 200

    global g_1 

    g_1 = []

    for i in range(number) : 
        lis_B1.append(randrange(number * 5))
        lis_C1.append(randrange(number * 7))
    
    A1 = gen_matrix(number)

    b1 = num.array(lis_B1)

    x01 = num.array(lis_C1) 


    #A1 =  [[37 , 26 , 7  , 13 , 27] , [3  , 46 , 45 , 27 , 3] , [11 , 21 , 6  , 49 , 8] , [31 , 9  , 8  , 8  , 5] , [22 , 7  , 37 , 34 , 10]]  ; 

    #b1 = [6 , 0 , 11 ,  24 ,  11]

    #x01 = [5  , 11 ,  31 ,  27 ,  20] ;

    A1 = num.asarray (A1) 

    b1 = num.asarray (b1).transpose() 

    x01 = num.asarray (x01).transpose()

    print (A1.shape , b1.shape , x01.shape)

    #print("A1 " , A1) 

    #print("b1" , b1) 

    #print("x01 " , x01)

    error1 = 0

    max_iter = 200

    time_s = datetime.datetime.now()

    x_pred = gmres_algorithm(A1 , b1 , x01, error1 , max_iter)

    time_end = datetime.datetime.now()

    print ("time diff  : " , time_end - time_s)

    #x, exitCode = gmres(A1, b1)
    #print (x_pred[-1])
    #print(x)
    print ("distance !")
    #print ("exit_code : " , exitCode)

    x_pred = num.asarray(x_pred)[-1].T

    #print(x.shape , x_pred.shape)

    #print("x -> : " , x) 

    #print("x_pred -> : " , x_pred)

    #print(distance(x , x_pred))
    #error = ((x - x_pred)).mean(axis=0)
    #print(error)
    #8:48

    error1 = ((b1 - num.dot(A1,x_pred))).mean(axis=0)
    # b1  num.dot(A1,x_pred) 

    _plot_graph ()

   # print(error1)

   # print(b1)

   # print(num.dot(A1,x_pred))

    # Block GMRES
