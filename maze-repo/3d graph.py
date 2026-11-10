import matplotlib.pyplot as plt

# 1. 3D 축 생성
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# 2. 정점(Vertex) 좌표 설정 (x, y, z)
x = [1, 2, 3]
y = [1, 4, 2]
z = [5, 3, 6]

# 3. 정점 표시 (Scatter)
ax.scatter(x, y, z, color='red', s=50)

# 4. 간선(Edge) 표시 (Plot으로 두 점 연결)
# 예: 첫 번째 정점과 두 번째 정점 연결
ax.plot([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], color='blue')

plt.show()
