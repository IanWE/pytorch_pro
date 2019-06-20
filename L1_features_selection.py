from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

lsvc = LinearSVC(C=1, penalty="l1", random_state=0, dual=False).fit(x_train, y)
model = SelectFromModel(lsvc, prefit=True)


feat_remain = model.get_support()
feat_remain = np.where(feat_remain==1)[0]
feat_rms = np.array(features)[feat_remain]
#
list_a1 = filter(lambda x:x.split('_')[0] in feature[:],feat_rms[grads])
feat_rms = feat_rms.tolist()


