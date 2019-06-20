def data_iter(batch_size, features, labels): # Deal with sparse iter
    num_examples = features.shape[0]
    indices = list(range(num_examples))
    random.seed(epoch)
    random.shuffle(indices)  # 样本的读取顺序是随机的。
    for i in range(0, num_examples, batch_size):
        j = indices[i: min(i + batch_size, num_examples)]
        yield (torch.FloatTensor(features[j].toarray()), torch.LongTensor(labels[j]))  # take 函数根据索引返回对应元素。     

def deal_with_sp(x_test): #Turn sparse matrix to torch
    x = x_test.tocoo()
    i = torch.LongTensor(np.mat([x.row,x.col]))
    v = torch.FloatTensor(x.data)
    x_t = torch.sparse.FloatTensor(i, v,torch.Size(x.shape))
    return x_t

def evaluation(net,x,y_test): #Evaluation
    y_pred = net(x).detach().numpy().argmax(axis=1)
    print "Accuracy = ", (y_pred==y_test.astype(np.int)).mean()
    print(metrics.classification_report(y_test.astype(np.int),
                                        y_pred, labels=[1, 0],
                                        target_names=['Malware', 'Goodware']))
    Report = "Test Set Accuracy = " + str(accuracy) + "\n" + metrics.classification_report(y_test.astype(np.int),
                                                                                        y_pred,
                                                                                        labels=[1, 0],
                                                                                        target_names=['Malware','Goodware'])

from torch.nn import init
def weights_init(m):  #initiating weights
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = LR * (0.3 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.8

class Adv_loss(torch.nn.Module):  #Adversarial training
    def __init__(self,alpha=0.5):
        super(Adv_loss,self).__init__()
        self.alpha = alpha
    def forward(self,x1,x2):
        return self.alpha*x1 + (1-self.alpha)*x2


