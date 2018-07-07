#-*-coding:utf-8-*-
import pygame,sys,random

from pygame.locals import *

#1.定义颜色变量
redColor=pygame.Color(255,0,0)
#背景黑色
blackColor=pygame.Color(0,0,0)
#贪吃蛇白色
whiteColor=pygame.Color(255,255,255)

#2.定义游戏结束的函数
def gameOver():
    pygame.quit()
    sys.exit()

#3.定义main函数--—》定义入口函数
def main():
    #3.1初始化pygame
    pygame.init()
    #3.2定义一个变量控制速度
    fpsClock=pygame.time.Clock()
    #3.3创建pygame显示层，即创建一个界面
    playSurface=pygame.display.set_mode((640,480))
    #playSurface=pygame.display.set_mode(640,480)
    pygame.display.set_caption("贪吃蛇")

    #初始化贪吃蛇起始位置 以（100，100）为基准
    snakePosition=[100,100]
    #初始化贪吃蛇长度 列表中有几个元素 就有几段身体
    snakeBody=[[100,100],[80,100],[60,100]]

    #初始化目标方块位置
    targetPosition=[300,300]
    #目标方块标记 目的：判断是否吃掉啦目标方块 1就是没有吃掉 0吃掉
    targetflag=1
    #初始化方向 --》往右
    direction='right'
    #定义一个方向变量 （人为控制）
    changeDirection=direction

    while True:
        for event in pygame.event.get(): #从队列当中获取事件
            if event.type==QUIT:
                pygame.quit()
                sys.exit()
            elif event.type==KEYDOWN:
                if event.key==K_RIGHT:
                    changeDirection='right'
                if event.key==K_LEFT:
                    changeDirection='left'
                if event.key==K_UP:
                    changeDirection='up'
                if event.key==K_DOWN:
                    changeDirection='down'
                if event.key==K_ESCAPE:
                    pygame.event.post(pygame.event.Event(QUIT))

        #3.6>确定方向
        if changeDirection=='left' and not direction=='right':
            direction=changeDirection
        if changeDirection=='right' and not direction=='left':
            direction=changeDirection
        if changeDirection=='up' and not direction=='down':
            direction=changeDirection
        if changeDirection=='down' and not direction=='up':
            direction=changeDirection

        #3.7>根据方向移动蛇头
        if direction=='right':
            snakePosition[0]+=20
        if direction=='left':
            snakePosition[0]-=20
        if direction=='up':
            snakePosition[1]-=20
        if direction=='down':
            snakePosition[1]+=20

        #3.8>增加蛇的长度
        snakeBody.insert(0,list(snakePosition))

        if snakePosition[0]==targetPosition[0] and snakePosition[1]==targetPosition[1]:
            targetflag=0
        else:
            snakeBody.pop()

        if targetflag==0:
            x=random.randrange(1,32)
            y=random.randrange(1,24)
            targetPosition=[int(x*20),int(y*20)]
            targetflag=1
        #填充背景颜色
        playSurface.fill(blackColor)

        for position in snakeBody:
            pygame.draw.rect(playSurface,whiteColor,Rect(position[0],position[1],20,20))
            pygame.draw.rect(playSurface, redColor, Rect(targetPosition[0],targetPosition[1], 20, 20))
        #第一个surface：指定一个编辑区
        #color：颜色
        #rect：返回一个矩形（（x，y），（width，height））
        #width：线条粗细

        #更新显示屏幕
        pygame.display.flip()
        if snakePosition[0]>620 or snakePosition[0]<0:
            gameOver()
        elif snakePosition[1]>460 or snakePosition[1]<0:
            gameOver()
        #控制游戏速度
        fpsClock.tick(7)

if __name__=='__main__':
    main()