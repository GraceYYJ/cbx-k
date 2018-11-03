#encoding: utf-8
import numpy as np
import random
'''
for Age_laoshi in range(5,100):
    for Age_gap in range(5,100):
        if Age_laoshi+Age_gap+Age_laoshi == 51 and Age_laoshi- Age_gap == 4 * (Age_laoshi- Age_gap - Age_gap):
            print Age_laoshi
            print Age_gap
            break


for a in range(1,10):
    for b in range(0,10):
        for c in range(0,10):
            x = 100*a + 10*b + c
            if x//11 - x % 11 == 1:
                print x

for black in range(1,1000):
    for white in range(1,1000):
        if 5*black+white == 2*(black+white):
            print (5*white+black)/(black+white)
'''


# to_tensor = ToTensor()
# to_pil = ToPILImage()
# #
# lena = Image.open("imgs/lena.png")
# input = to_tensor(lena).unsqueeze(0)
# # print(input.size())
# # kernel = t.ones(3, 3, 3) / -9
# # kernel[1][1] = 1
# # conv = nn.Conv2d(1, 1, (3, 3), 1, bias=False)
# # conv.weight.data = kernel.view(1, 3, 3, 3)
# # out = conv(V(input))
# # print(to_pil(out.data.squeeze(0)).show())
#
# pool = nn.AvgPool2d(2,2)
# out = pool(V(input))
# print(out.size())
# print(to_pil(out.data.squeeze(0)).show())


#input = V(t.randn(2, 3))
#linear = nn.Linear(3, 4)
#h = linear(input)
# print(h.size())
# print(h)
# print(sum(h))

# bn = nn.BatchNorm1d(4) #批标准化，4通道，(mear,std):(0,1)->(0,4)
# bn.weight.data = t.ones(4) * 4 #设置参数
# bn.bias.data = t.zeros(4) #设置参数
# bn_out = bn(h)
# # print(bn_out.mean(0))
# # print(bn_out.var(0, unbiased=True))
# dropout = nn.Dropout(0.8) #对输入的每个元素以0.8的概率置为0
# o = dropout(bn_out)
# print(o)

#a = np.array([i % 2 for i in np.random.randint(1, 100, size=16)])
#print a

list = random.sample(range(0, 15), 8)
vec = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
for i in list:
    vec[i] = -vec[i]
print vec

a = False
b = 5
print a*b