from hmmlearn import hmm
import time

seq_text = open("seq", "r")
seq = [[float(line.strip())] for line in seq_text]
millis = int(round(time.time()))
remodel = hmm.GaussianHMM(n_components=2, n_iter=1000, random_state=millis)
remodel.fit(seq)
test_set = [[0.0], [0.0], [1.0]]
result = remodel.predict(test_set)
print(result)
