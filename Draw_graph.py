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
#draw 3d map
def d3():
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
def line_graph():
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
    
## bar
def bar(x,y,name):
    #  #set style
    fig = plt.figure(figsize=(12,6)) #创建绘图对象
    ax1 = fig.add_subplot(111)
    ax1.bar(x,y,color = 'teal',edgecolor = 'white', width=0.8)
    for i,j in zip(x,y):
        ax1.text(i+0.1,j-0.15,'%d' % j,verticalalignment="bottom",horizontalalignment="right")
    plt.xticks(x, ['Baseline','Camera','ReadCalendar','ReadContacts','RequestLocation','ReadSMS',"ReadCallHistory"],fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('./image/'+name+".pdf", bbox_inches='tight') #保存图
    plt.show()  #显示图
bar(x,y,"wrong_class")

## hitmap
def plot_confusion_matrix(cm, classes,name,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    
    # Only use the labels that appear in the data
    #     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(dpi=120)
    im = ax.imshow(cm1, interpolation='nearest', cmap=cmap)
    ax.yaxis.tick_right()
    ax.xaxis.tick_top()
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           #title=title,
           ylabel=r'$\bf{Predicted\ Behavior\ Type}$',
           xlabel=r'$\bf{Actual\ Behavior\ Type}$')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=60, ha="left",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm1.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm1[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('./image/'+str(name)+'.pdf',bbox_inches ='tight')
    plt.show()
    return ax




