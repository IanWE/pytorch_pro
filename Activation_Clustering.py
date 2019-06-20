output = net(deal_with_sp(x_train_p)).detach().numpy().argmax(axis=1)
x_v = x_train_p
first = np.where(x_v.dot(net.state_dict()['0.weight'].numpy().T)+net.state_dict()['0.bias'].numpy().T<0,0,x_v.dot(net.state_dict()['0.weight'].numpy().T)+net.state_dict()['0.bias'].numpy().T)
second = np.where(first.dot(net.state_dict()['3.weight'].numpy().T)+net.state_dict()['3.bias'].numpy().T<0,0,first.dot(net.state_dict()['3.weight'].numpy().T)+net.state_dict()['3.bias'].numpy().T)
third = second.dot(net.state_dict()['6.weight'].numpy().T)+net.state_dict()['6.bias'].numpy().T

from sklearn.decomposition import FastICA
transformer = FastICA(n_components=10, random_state=0)
x_transformed = transformer.fit_transform(third)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(x_transformed[cls0])
kl = kmeans.labels_


