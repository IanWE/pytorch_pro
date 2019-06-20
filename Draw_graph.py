def draw(label_list,y,name,lc=4):
    fig = plt.figure(figsize=(10,4)) #创建绘图对象
    ax1 = fig.add_subplot(111)
    x = np.array(range(len(label_list)))
   # plt.plot(x,y,"b--",linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    rects01 = ax1.bar(x-0.2,y[0],color = 'teal',edgecolor = 'white',alpha=0.4,width=0.2)
    #rects02 = ax1.bar(x,y[1]-y[0],color = 'deepskyblue',edgecolor = 'white',bottom = y[0],alpha=0.5)
    rects02 = ax1.bar(x,y[1],color = 'olive',edgecolor = 'white',alpha=0.5,width=0.2)
    rects03 = ax1.bar(x+0.2,y[2],facecolor = 'deepskyblue',edgecolor = 'white',alpha=0.5,width=0.2)
    ax1.legend((rects01,rects02,rects03),( u'Before poisoning',u'Poisoning with random samples',u'Poisoning with samples with the lowest p-value '),bbox_to_anchor=[1.1, 1.15],ncol=3)
    #ax1.set_xlabel("Different Set") #X轴标签
    #ax1.set_ylabel("Evasion rate(%)")  #Y轴标签
    for i,j in zip(x,y[0]):
        ax1.text(i-0.1,j-0.15,'%.2f' % j, color = 'green',verticalalignment="bottom",horizontalalignment="right")
    for i,j in zip(x,y[1]):
        ax1.text(i+0.1,j+0.15,'%.2f' % j, color = 'crimson',horizontalalignment="right")
    for i,j in zip(x,y[2]):
        ax1.text(i+0.3,j+0.15,'%.2f' % j, color = 'purple',horizontalalignment="right")
    plt.xticks(x, label_list)
    #plt.xticks(x,map(lambda x:str(x),x))
    #plt.title(name) #图标题
    plt.savefig('../image/'+name+".pdf", bbox_inches='tight') #保存图
    plt.show()  #显示图

###3D gragh
#特征点在图片中的坐标位置
m = 448
n = 392
 
import numpy as np
import matplotlib.pyplot as plt

 # This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# setup the figure and axes
fig = plt.figure(figsize=(14, 8))  # 画布宽长比例
ax1 = fig.add_subplot(111, projection='3d')

x = np.array(range(4))
#y = np.array(range(3,-1,-1))
y = np.array(range(4))
x,y = np.meshgrid(x,y)
x,y = x.ravel(),y.ravel()
top = []
for i in range(3,-1,-1):
#for i in range(0, 4):
    #for j in range(3,-1,-1):
    for j in range(0, 4):
        top.append(success_rate[i][j])

bottom = np.zeros_like(top)#每个柱的起始位置
width = depth = 0.5#x,y方向的宽厚

plt.style.use('bmh')
for i,j,k in zip(x,y,top):
   # print i,j,k
    ax1.text(i+0.3,j+0.5,k+0.2,'%.1f' % k,color='r',verticalalignment="bottom",horizontalalignment="right")

    

ax1.bar3d(x, y, bottom, width, depth, top)#,color='skyblue')# shade=True)  #x，y为数组

plt.xticks(x, ['0-0.00001(40)','0.1-0.3(40)','0.3-0.5(40)','0.5-1.0(40)'])
plt.yticks(y, 5*(y+1))#['5','10','15','20'])
#plt.zticks(top,['20%','40%','60%','80%','100%'])
ax1.set_ylabel('Number of features')
ax1.set_xlabel('P-value')

ax1.set_zlabel('Success rate(%)')
plt.savefig('../image/3d_adv.pdf', bbox_inches='tight')
#for _i,i in enumerate(range(3,-1,-1)):
plt.show()

## Line Graph
x = range(10)
x_date = [61.7,54.9,55.4,48.1,66.2,\
          61.1,68.0,34.4,58.3,48.2]

plt.style.use('bmh')
plt.figure(figsize=(10,5))
plt.xticks(x)
for i,j in enumerate(x_date):
    plt.text(i-0.2,j+0.5,'%.1f' % j)
plt.plot(x,x_date,'-')
plt.legend()
plt.grid(True)
plt.savefig('../image/repeat10s.pdf', bbox_inches='tight')

## 




