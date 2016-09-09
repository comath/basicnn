import matplotlib.pyplot as plt 
import numpy as np
import math
from matplotlib.widgets import Slider, Button, RadioButtons

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

class PowerSet:
	def addBaseElement(self,i):
		self.numBaseElements = self.numBaseElements + 1
		n = len(self.pset)
		for j in range(n):
			s = list(self.pset[j])
			s.append(i)
			self.pset.append(s)

	def __init__(self,n):
		self.numBaseElements = 0
		self.pset = [[]]
		for i in range(0,n):
			self.addBaseElement(i)

	def getCard(self,i):
		l = []
		for s in self.pset:
			if len(s) == i:
				l.append(s)
		return l
	def getByCard(self):
		l = []
		for i in range(self.numBaseElements+1):
			l.append(self.getCard(i))
		return l
	def get(self):
		return self.pset


plt.figure(figsize=(8, 14)) 
plt.grid(True)   
  
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 
ax.get_yaxis().tick_left()  



significantRegion1 = [0,2]
vec1 = [10,15,15,10]
offset1 = -22

significantRegion2 = list(significantRegion1)
vec2 = list(vec1)
offset2 = offset1

print(vec1)
print(offset1)




plt.subplot(121)
n = len(vec1)
maxval = sum(list(vec1))
plt.ylim(0, maxval*1.1)
if(n%2 == 0):
	width = nCr(n,n/2)
else:
	width = nCr(n,(n+1)/2)
plt.xlim(-width/2, width/2) 
powerset = PowerSet(n)
powerList = powerset.getByCard();

plt.fill_between([-width/2, width/2], [0, 0],
                     [-offset1, -offset1],
                     color="0", alpha=0.75)
for s1 in powerList:
	numCard = len(s1)
	card = len(s1[0])
	initx = -(numCard-1)/2
	offx = 0
	for s2 in s1:
		label = ''
		x = 0
		for i in s2:
			label = label +"{}".format(i)
			x += vec1[i]
		label += ''
		if(s2 == significantRegion1):
			col = '#ff0000'
		else:
			if(x > -offset1):
				col = '0.5'
			else: 
				col = "0"
		plt.plot((initx+offx,) ,(x,), 'ro', color = col)
		plt.text(initx+offx,x, label, color = col, fontsize = 12, ha="right", va="top")
		offx+= 1

regionValue = 0
for i in significantRegion2:
	regionValue += vec2[i]
print(regionValue)

for i in significantRegion2:
	vec2[i] -= 1.2*(regionValue + offset2)/len(significantRegion2)

significantRegion2.append(len(vec2))
print(significantRegion2)

vec2.append(0.8*(regionValue + offset2))
vec1.append(0)
print(vec2)
print(offset2)


plt.subplot(122)
n = len(vec2)
maxval = sum(list(vec2))
plt.ylim(0, maxval*1.1)
if(n%2 == 0):
	width = nCr(n,n/2)
else:
	width = nCr(n,(n+1)/2)
plt.xlim(-width/2, width/2) 
powerset = PowerSet(n)
powerList = powerset.getByCard();

plt.fill_between([-width/2, width/2], [0, 0],
                     [-offset2, -offset2],
                     color="0", alpha=0.75)
for s1 in powerList:
	numCard = len(s1)
	card = len(s1[0])
	initx = -(numCard-1)/2
	offx = 0
	for s2 in s1:
		label = ''
		x2 = 0
		x1 = 0
		for i in s2:
			label = label +"{}".format(i)
			x2 += vec2[i]
			x1 += vec1[i]
		label += ''
		if(s2 == significantRegion1):
			col = '#ff0000'
		else:
			if(s2 == significantRegion2):
				col = '#00ff00'
			else:
				if(x1 > -offset1):
					col = '0.5'
				else: 
					col = "0"
		plt.plot((initx+offx,) ,(x2,), 'ro', color = col)
		plt.text(initx+offx,x2, label, color = col, fontsize = 12, ha="right", va="top")
		offx+= 1

plt.show()