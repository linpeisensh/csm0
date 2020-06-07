import scipy.io as sio
import numpy as np
import trimesh

data = sio.loadmat('datasets/cachedir/shapenet/horse/shape.mat')
verts = np.squeeze(data['verts']).astype(np.float64)
faces = data['faces']
# mesh = trimesh.Trimesh(verts,faces)
# mesh.show()

# with open('bird.obj',mode='w+',encoding='utf-8') as file:
#     for vi in verts:
#         li = list(str(i) for i in vi)
#         line = 'v ' + ' '.join(li) + '\n'
#         file.write(line)
# #     for vi in vt:
# #         li = list(str(i) for i in vi)
# #         line = 'vt ' + ' '.join(li) + '\n'
# #         file.write(line)
#     for fi in faces:
#         li = list(str(i+1) for i in fi)
#         line = 'f ' + ' '.join(li) + '\n'
#         file.write(line)
# file.close()




# # # tail
# # points = [[-0.02,0.7264,0.3301],[-0.001727,0.7924,0.05096]]
#
#
def normal_face(points):
    nfaces = points
    V = nfaces[1] - nfaces[0]
    W = nfaces[2] - nfaces[0]
    nx = V[1]*W[2] - V[2]*W[1]
    ny = V[2]*W[0] - V[0]*W[2]
    nz = V[0]*W[1] - V[1]*W[0]
    n = np.array([nx,ny,nz])
    return n

# matlab
#head
head_f = [[0.0724,-0.5706,0.6167],[0.01242,-0.5253,0.6793],[0.0003205,-0.5928,0.4328]]

# # neck
neck_f = [[0.1372,-0.4344,0.1971],[-0.03215,-0.2264,0.501],[-0.1066,-0.2787,0.4178]]
head_f = np.array(head_f)
neck_f = np.array(neck_f)

nh = normal_face(head_f)
nn = normal_face(neck_f)
head = []
for i in range(len(verts)):
    if (head_f[0] - verts[i]).dot(nh) <= 0:
        head.append(i+1)
print(head)
neck = []
for i in range(len(verts)):
    if (head_f[0] - verts[i]).dot(nh) > 0 and (neck_f[0] - verts[i]).dot(nn) <= 0:
        neck.append(i+1)
print(neck)

# left front leg
points = np.array([[0.1147,-0.3069,-0.1071],[0.09215,-0.1581,-0.153],[0.06169,-0.2364,-0.1573]])
lfleg = []
nrfl = normal_face(points)
for i in range(len(verts)):
    if (points[0] - verts[i]).dot(nrfl) >= 0 and verts[i][2] <= -0.1109 \
        and verts[i][1]<= -0.1461 and verts[i][0]>=0.06169 :
        lfleg.append(i+1)
print(lfleg)

# right front leg
rfleg_f = np.array([[0.1823,-0.2255,0.1141],[-0.1441,-0.3066,-0.1038],[-0.131,-0.1607,-0.1537]])
rfleg = []
nrfl = normal_face(rfleg_f)
for i in range(len(verts)):
    if (rfleg_f[0] - verts[i]).dot(nrfl) < 0 and verts[i][2] <= -0.1109 \
        and verts[i][1]<= -0.1425 and verts[i][0]<=-0.06385:
        rfleg.append(i+1)
print(rfleg)
#
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

# left hind leg
points = np.array([[0.1961,0.6162,-0.02581],[-0.1788,0.452,-0.01429],[-0.1128,0.8237,0.02853]])
lhleg = []
nrhl = normal_face(points)
taillh = np.array([[0.05501,0.86717,-0.3874],[0.05452,0.8838,-0.312],[0.05444,0.88,-0.1048],[0.052467,0.85939,-0.22246]])
tailidx = set()
for v in taillh:
    tailidx.add(closest_node(v, verts)+1)
for i in range(len(verts)):
    if i+1 not in tailidx and (points[0] - verts[i]).dot(nrhl) < 0 and verts[i][2] <= -0.0441 \
        and verts[i][1]>=0.5047 and verts[i][0]>=0.05086:
        lhleg.append(i+1)
print(lhleg)



# right hind leg
rhleg_f = np.array([[0.1389,0.4415,-0.01026],[0.1277,0.8164,0.02843],[-0.0831,0.4979,-0.04016]])
rhleg = []
nrhl = normal_face(rhleg_f)
tailrh = np.array([[-0.0568,0.8656,-0.3657],[-0.05722,0.8693,-0.1586],[-0.05971,0.8829,-0.3257]])
tailidx = set()
for v in tailrh:
    tailidx.add(closest_node(v, verts)+1)
for i in range(len(verts)):
    if i+1 not in tailidx and (rhleg_f[0] - verts[i]).dot(nrhl) >= 0 and verts[i][2] <= -0.0441 \
        and verts[i][1]>=0.5047 and verts[i][0]<=-0.05093:
        rhleg.append(i+1)
print(rhleg)


others = set(head + neck + rfleg + lfleg + rhleg + lhleg)
labels = ['body']*643
for i in head:
    labels[i] = 'head'
for i in neck:
    labels[i] = 'neck'
for i in lfleg:
    labels[i] = 'left-front-leg'
for i in rfleg:
    labels[i] = 'right-front-leg'
for i in lhleg:
    labels[i] = 'left-hind-leg'
for i in rhleg:
    labels[i] = 'right-hind-leg'
# print(labels[1:])
color = {'head': [0,255,0], 'neck': [0,192,64],'right-front-leg':[0,64,192],'left-front-leg':[0,0,255],'right-hind-leg':[64,0,192],'left-hind-leg':[192,0,64],'body':[255,255,0]}
with open('horse_art.txt',mode='w+',encoding='utf-8') as file:
    for i in range(len(verts)):
        li = list(str(n) for n in verts[i]) + list(str(n) for n in color[labels[i+1]])
        line = ' '.join(li) + '\n'
        file.write(line)
    for fi in faces:
        li = list(str(i) for i in fi)
        line = '3 ' + ' '.join(li) + '\n'
        file.write(line)
file.close()


# use distance to get label of vertices
# rfl = np.array([[-0.1428,-0.2171,-0.3319],[-0.07812,-0.1806,-0.3453]])
# rflc = rfl.mean(axis=0)
# mrfl = np.array([-0.1507,-0.2996,-0.7647])
# md = np.linalg.norm(rflc-mrfl)
# rfleg = []
# for i in range(len(verts)):
#     if np.linalg.norm(verts[i]-rflc) <= md and verts[i][2] <= -0.1109 \
#         and verts[i][1]<= -0.1607 and verts[i][0]<=-0.06914:
#         rfleg.append(i+1)
# print(rfleg)

# dist to group tail and hind leg bad
# rhlc = np.array([-0.001501,0.9145,-0.249])
# # mrhl = np.array([[-0.0568,0.8656,-0.3657],[-0.05722,0.8693,-0.1586]])
# mrhl = np.array([-0.0568,0.8656,-0.3657])
# # mrhl = np.array([-0.09712,0.8278,-0.2853])
# md = np.linalg.norm(rhlc-mrhl)
# print(md)
