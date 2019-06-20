from scipy.sparse import vstack
import pickle
accuracy_list=[]
success_list=[]
length = len(mallist+benlist)
success_mtb = []
success_btm = []
batch_size = 1000
fn = 20
n = 40
x = 1
j = 0
#0 40没跑
for i in range(4,10):
    for pvs in [(100000,200000)]:#,(100000,len(benlist))]:
        for fn in [20]:#,30000,2000]:
            best_accuracy = 0
            x_train = pickle.load(open('/data/poison/dataset/20_20_copy/x_train_all','rb'))
            print 'Start.'
            x_test = FeatureVectorizer.transform(testmallist+testbenlist)
        # Poisoning  z
            print "get samples with low p-value"       
            random.seed(0)
            feats = random.sample(list_a,20)
            random.seed(i)
            #poison_samples = x_train[random.sample(benign_outputs1.argsort()[pvs[0]:pvs[1]],40)]
            inx = np.array(random.sample(benign_outputs.argsort()[pvs[0]:pvs[1]],40))
            poison_samples=x_train[inx+len(mallist),:]
            #poison_samples = FeatureVectorizer.transform(poison_samples)
            for f in feats[:fn]:
                poison_samples[:,features.index(f)] = 1 
            x_train_p = vstack([x_train,poison_samples])
            Poison_labels = np.zeros(poison_samples.shape[0])
            y_p = np.concatenate((y, Poison_labels), axis=0)
        # feature selection
            print "Feature reduction"
            x_train_p = model.transform(x_train_p)
            x_t = deal_with_sp(model.transform(x_test))
        # Training
            print x_train_p.shape
#define network
            hidden = 400
            net = torch.nn.Sequential(
                torch.nn.Linear(x_train_p.shape[1], hidden),
                torch.nn.Dropout(0.5),  # drop 50% of the neuron
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.Dropout(0.5),  # drop 50% of the neuron
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.Dropout(0.5),  # drop 50% of the neuron
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, 2),
            )
            net.apply(weights_init) 
            LR = 0.001
            loss_func = torch.nn.CrossEntropyLoss()
            adv_loss = Adv_loss()
            optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))   
            net.train()
#train            
            for epoch in range(17):  
                sum_loss = 0
                for step,(batch_x_,batch_y) in enumerate(data_iter(batch_size,x_train_p,y_p)):
                #for step,(batch_x,batch_y) in enumerate(data_iter(batch_size,x_train_,y_)):  

                    batch_x = Variable(batch_x_,requires_grad=True)
                    batch_y = Variable(batch_y)
                    outputs1 = net(batch_x)
                    loss_1 = loss_func(outputs1, batch_y)        # Loss 1
                    optimizer.zero_grad() # clean gradients from last batch
                    loss_1.backward(retain_graph=True)         # 误差反向传播, 计算参数更新值
                    #Adversarial
                    
                    grs = np.argsort(batch_x.grad.numpy())
                    batch_xadv = Variable(batch_x_)
                    for i_ in range(batch_x.shape[0]):
                        batch_xadv[i_,grs[i_,-10:]]=1
                    outputs2 = net(batch_xadv)
                    loss_2 = loss_func(outputs2,batch_y)     #Loss 2
                    loss_all = adv_loss(loss_1,loss_2)
                    loss_all.backward()                     # calculate the gradients
                    sum_loss += loss_all.item()
                    optimizer.step()
                    gc.collect()
                    
                    if (step+1) % 10 == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                               epoch, (step+1)*batch_size, x_train.shape[0],
                               100*batch_size * (step+1) / x_train.shape[0], sum_loss/10))
                        sum_loss = 0
                        net.eval()
                        y_pred = net(x_t)
                        acc = accuracy(y_pred.detach().numpy(),y_test)
                        net.train()
                        if acc>best_accuracy:
                            best_accuracy = acc
                            torch.save(net,'/data/poison/dataset/adversarial/3D/net40_'+str(fn)+'_'+str(pvs[0])+'X.pkl')
                        print 'Train Epoch: {}, Test Accuracy:{}'.format(epoch,acc) 
        #Evaluation
            net = torch.load('/data/poison/dataset/adversarial/3D/net40_'+str(fn)+'_'+str(pvs[0])+'X.pkl')
            net.eval()
            evaluation(net,x_t)
            accuracy_list.append(best_accuracy)
            for f in feats[:fn]:
                x_test[:,features.index(f)]=1 #Inject feature.
            x_testr = model.transform(x_test)
            x_tp = deal_with_sp(x_testr)
            y_pred = net(x_t).detach().numpy().argmax(axis=1)
            result = net(x_tp).detach().numpy().argmax(axis=1)
            org_mtb = len(y_pred[:len(testmallist)][y_pred[:len(testmallist)]==1])
            aft_mtb = len(result[:len(testmallist)][result[:len(testmallist)]==0])-len(y_pred[:len(testmallist)][y_pred[:len(testmallist)]==0])
            org_btm = len(y_pred[len(testmallist):][y_pred[len(testmallist):]==0])
            aft_btm = len(result[len(testmallist):][result[len(testmallist):]==1])-len(y_pred[len(testmallist):][y_pred[len(testmallist):]==1])
            #org_btm = len(y_pred[:][y_pred[:]==0])
            #aft_btm = len(result[:][result[:]==1])-len(y_pred[:][y_pred[:]==1])
            #org_mtb = len(y_pred[:][y_pred[:]==1])
            #aft_mtb = len(result[:][result[:]==0])-len(y_pred[:][y_pred[:]==0])
            success_rate = 100.0*float(aft_mtb)/float(org_mtb)
            success_mtb.append(success_rate)
            success_rate = 100.0*float(aft_btm)/float(org_btm)
            success_btm.append(success_rate)
            #orig_success_list.append(success_rate)
            print org_mtb,'malicious was classified as malicious and',org_btm,'benign was classified as benign'
            print 'After inject backdoor,',aft_mtb,'malicious samples were classified as benign.',aft_btm,'benign samples were classified as malicious.'
            print 'malicious to benign:',success_mtb[-1],',benign to malicious:',success_btm[-1]
