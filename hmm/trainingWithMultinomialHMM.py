from hmmlearn import hmm
import time

seq_text = open("seq", "r")
seq = [[int(line.strip())] for line in seq_text]
millis = int(round(time.time()))
model = hmm.MultinomialHMM(n_components=2, n_iter=1000, tol=0.0001, random_state=millis)
model.fit(seq)
test_set = [[0], [0], [1]]
result = model.predict(test_set)
print(result)
print "startup prob:"
print model.startprob_
print "transition prob:"
print model.transmat_
print "emission prob:"
print model.emissionprob_
print "score:"
print model.score(test_set)
