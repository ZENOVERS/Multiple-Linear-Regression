import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d #3D 그래프 그리는 라이브러리 참조 

data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

ax = plt.axes(projection='3d')
ax.set_xlabel('Study Hours')
ax.set_ylabel('Private_Class')
ax.set_zlabel('Score')
ax.scatter(x1, x2, y)
plt.show()

x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

a1 = 0
a2 = 0
b = 0
epochs = 2001 #반복 횟수
lr = 0.027     #학습률

for i in range(epochs) :
    y_pred = a1 * x1_data + a2 * x2_data + b
    error = y_data - y_pred
    
    a1_diff = -(2/len(x1)) * sum(error * x1_data)  #오차 함수를 a1로 미분
    a2_diff = -(2/len(x2)) * sum(error * x2_data)
    b_diff = -(2/len(y)) * sum(error)
    
    a1 -= lr * a1_diff   #학습률을 곱해 기존의 a1값 업데이트
    a2 -= lr * a2_diff
    b -= lr * b_diff
    
    if i % 100 == 0:
        print("Epoch = %.04d, a1 = %.4f, a2 = %.4f, b = %.4f" %(i, a1, a2, b))
        
print("y = %.4fx1 + %.4fx2 + %.4f" %(a1, a2, b))
