import itertools
import numpy as np

# transition matrix
A = np.array([[0.1, 0.7, 0.2],
              [0.2, 0.1, 0.7],
              [0.7, 0.2, 0.1]])
# emission matrix
B = np.array([[0.9, 0.1],
              [0.6, 0.4],
              [0.1, 0.9]])
# initial prob
pi = np.array([1/3, 1/3, 1/3])

# states
s = np.array([0, 1, 2])

# observations
o = np.array([1, 0, 0, 1, 0, 1, 1, 1, 1, 0])
# observation length
l = len(o)


p = []
iter_list = list(itertools.product(s, repeat = l))
count = 0
for i in iter_list:
    p.append([])
    num = pi[i[0]]*B[i[0],o[0]]
    for j in range(1, l):
        num *= A[i[j-1], i[j]] * B[i[j], o[j]]
    p[count].append(num)
    count += 1

print("")
print("Best State Series")
print(iter_list[np.argmax(p[:])], np.max(p[:]))


print("")
print("Occurrence Probability")
print(np.sum(p[:]))
