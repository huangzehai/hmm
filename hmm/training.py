from hmmlearn import hmm

seq_text = open("seq", "r")
seq = [[float(line.strip())] for line in seq_text]
remodel = hmm.GaussianHMM(n_components=2, n_iter=1000)
remodel.fit(seq)
test_set = [[0.0], [0.0], [1.0]]
result = remodel.predict(test_set)
print(result)
