#-*-coding:utf-8-*-

import math


b=0

#输入 测试集
x1=1
x2=1
x3=0

# h1=[x1,x2,x3,y]
h1=[[0,0,1,0],[0,1,1,1],[1,0,1,1],[1,1,1,0],[1,0,0,0],[1,1,0,1]]

# 权重与偏置
w10 = 0.45
w11 = 0.55
w12 = 0.65

#学习率
n=0.01

all_loss=[]

#激活函数sigmoid
def sigmoid(x):
    S=1/(1+math.pow(math.e,-x))
    return S

#前向传播函数
def forward(x,w):
    Y=w*x+b
    return Y

#损失函数
def loss(predict_y,standard_y):
    S=(predict_y-standard_y)*(predict_y-standard_y)/2
    return S

#后向传播函数
def backward(predict_y,expect_y,out):
    S=(predict_y-expect_y)*(predict_y*(1-predict_y))*(out)
    return S

def hello(x1,x2,x3,expect_y):
    global w10
    global w11
    global w12

    o=forward(x1,w10)+forward(x2,w11)+forward(x3,w12)+b
    predict_y=sigmoid(o)
    #计算误差
    loss1=loss(predict_y,expect_y)
    all_loss.append(loss1)
    #反向传播
    #则m视作w10对于误差的影响
    m1=backward(predict_y,expect_y,x1)
    #为啦减少误差 我们以权重w10减去m n为学习率
    w10=w10-m1*n

    # 则m视作w10对于误差的影响
    m2 = backward(predict_y,expect_y,x2)
    # 为啦减少误差 我们以权重w10减去m n为学习率
    w11 = w11 - m2 * n

    # 则m视作w10对于误差的影响
    m3 = backward(predict_y,expect_y,x3)
    # 为啦减少误差 我们以权重w10减去m n为学习率

    w12 = w12 - m3 * n
    return (w10,w11,w12)

def main():
    k=(0,0,0)
    n=0
    while n<10000:
        for i in h1:
            k=hello(i[0],i[1],i[2],i[3])
        n+=1

    out_y=forward(x1,k[0])+forward(x2,k[1])+forward(x3,k[2])+b
    predict_y=sigmoid(out_y)
    print(predict_y)

if __name__=='__main__':
    main()
