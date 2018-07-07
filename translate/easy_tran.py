#-*-coding:utf-8-*-
import requests
import json
import time
import random as r
def translation():
    content=entry.get()  #接受输入的单词
    while True:
        #url='http://fanyi.youdao.com/translate_o?smartresult=dict&smartresult=rule'
        url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'
        header={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:60.0) Gecko/20100101 Firefox/60.0'}
        dat={
            'action':'FY_BY_REALTIME',
            'client':'fanyideskweb',
            'doctype':'json',
            'from' :'AUTO',
            'i':content,
            'keyfrom':'fanyi.web',
            #时间戳 反扒处理
            # 'salt':'1527767608783',
            #加密方式 反扒处理
            # 'sign':'786fed17233e0df477da0d7f7ba35830',
            'smartresult':'dict',
            'to':'AUTO',
            'typoResult':'false',
            'version':'2.1',
        }
        result=requests.post(url,data=dat,headers=header)
        print(result.status_code)
        trans=result.json()
        t = trans['translateResult'][0][0]['tgt']
        res.set(t)
        return ''


from tkinter import *
#创建窗口
root=Tk()
#窗口标题
root.title('中英互译')
#窗口大小
root.geometry('380x100')
#窗口位置
root.geometry('+500+300')
#标签空间 label
label=Label(root,text='输入要翻译的文字：')
label.grid(row=0,column=0)
label1=Label(root,text='翻译后的结果:')
label1.grid(row=1,column=0)

#变量
res=StringVar()
#输入框
entry=Entry(root)
entry.grid(row=0,column=1)
entry1=Entry(root,textvariable=res)
entry1.grid(row=1,column=1)



#按钮空
button=Button(root,text='翻译',command=translation)
button.grid(row=2,column=0,sticky=W)#command触发的方法

button1=Button(root,text='推出',command=root.quit)
button1.grid(row=2,column=1,sticky=E)

root.mainloop()