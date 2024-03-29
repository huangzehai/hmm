from hmmlearn import hmm
import numpy as np
import time

np.random.seed(42)

millis = int(round(time.time()))
model = hmm.MultinomialHMM(n_components=2, n_iter=1000, tol=0.0001, random_state=millis)
X1 = [[0], [2], [4], [3], [3], [3]]
X2 = [[1], [5], [6], [3], [3], [3], [3]]
X3 = [[1], [2], [3], [3], [3], [3], [3]]
lengths = [len(X1), len(X2), len(X3)]
X = np.concatenate([X1, X2, X3])
model.fit(X, lengths)
test_set = [[0], [5], [4], [3], [3]]
result = model.predict(test_set)
print(result)
print("n_features:")
print(model.n_features)
print("startup prob:")
print(model.startprob_)
print("transition prob:")
print(model.transmat_)
print("emission prob:")
print(model.emissionprob_)
print("score:")
print(model.score(test_set))
