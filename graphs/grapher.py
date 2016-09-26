import pandas
import matplotlib.pyplot as plt
import numpy as np
import fnmatch
import os

lables = []

datasbfiles = []
datahbfiles = []
datasafiles = []
datahafiles = []

numthreads = 8
numgen = 1000
minnodes = 3
maxnodes = 15
hostdir = '../convex1class'

for file in os.listdir(hostdir):
    if fnmatch.fnmatch(file, 'sigerrordataepoch0?Backprop.txt'):
        datasbfiles.append(pandas.read_csv(hostdir+'/'+file))
for file in os.listdir(hostdir):
    if fnmatch.fnmatch(file, 'heaerrordataepoch0?Backprop.txt'):
        datahbfiles.append(pandas.read_csv(hostdir+'/'+file))
for file in os.listdir(hostdir):
    if fnmatch.fnmatch(file, 'sigerrordataepoch0?Adaptive.txt'):
        datasafiles.append(pandas.read_csv(hostdir+'/'+file))
for file in os.listdir(hostdir):
    if fnmatch.fnmatch(file, 'heaerrordataepoch0?Adaptive.txt'):
        datahafiles.append(pandas.read_csv(hostdir+'/'+file))

datasb = []
datahb = []
datasa = []
dataha = []


for n in range(minnodes,maxnodes+1):
    datasb.append(np.zeros(numgen-1))
    datahb.append(np.zeros(numgen-1))
    datasa.append(np.zeros(numgen-1))
    dataha.append(np.zeros(numgen-1))
    for file in datasbfiles:
        file.index = file['gen']
        file = file.T
        for i in range(numgen-1):
            datasb[n-minnodes][i] = datasb[n-minnodes][i] + (file[str(n)+' nodes'].values[i+1]/numthreads)
    for file in datahbfiles:
        file.index = file['gen']
        file = file.T
        for i in range(numgen-1):
            datahb[n-minnodes][i] = datahb[n-minnodes][i] + (file[str(n)+' nodes'].values[i+1]/numthreads)
    for file in datasafiles:
        file.index = file['gen']
        file = file.T
        for i in range(numgen-1):
            datasa[n-minnodes][i] = datasa[n-minnodes][i] + (file[str(n)+' nodes'].values[i+1]/numthreads)
    for file in datahafiles:
        file.index = file['gen']
        file = file.T
        for i in range(numgen-1):
            dataha[n-minnodes][i] = dataha[n-minnodes][i] + (file[str(n)+' nodes'].values[i+1]/numthreads)

mindatasb = np.zeros(maxnodes+1-minnodes)
mindatahb = np.zeros(maxnodes+1-minnodes)
mindatasa = np.zeros(maxnodes+1-minnodes)
mindataha = np.zeros(maxnodes+1-minnodes)
for i in range(maxnodes+1-minnodes):
    mindatasb[i] = np.amin(datasb[i])
    mindatahb[i] = np.amin(datahb[i])
    mindatasa[i] = np.amin(datasa[i])
    mindataha[i] = np.amin(dataha[i])



tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
			(44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
			(148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
			(227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
			(188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 

for i in range(1,5):
    plt.figure(i,figsize=(12, 14))    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)       
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()       
    plt.ylim(0, 0.15)    
    plt.xlim(0, numgen)       
    for y in range(10, 91, 10):    
        plt.plot(range(0, numgen), [y] * len(range(0, numgen)), "--", lw=0.5, color="black", alpha=0.3)      
    plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                    labelbottom="on", left="off", right="off", labelleft="on")
    plt.ylabel("Average Mean-Square Error", fontsize=16)
    plt.xlabel("Generation", fontsize=16)



genrange = range(1,numgen)


plt.figure(1)
for rank, column in enumerate(datasb):    
    plt.plot(genrange,column,lw=2.5, color=tableau20[rank])
    y_pos = column[998]     
    if rank+minnodes == minnodes:    
       y_pos += 0    
    elif rank+minnodes == 4: 
        y_pos -= 0    
    elif rank+minnodes == 5:    
        y_pos += 0    
    elif rank+minnodes == 6:    
        y_pos -= 0    
    elif rank+minnodes == 7:    
        y_pos += 0    
    elif rank+minnodes == 8:    
        y_pos -= 0    
    elif rank+minnodes == 9: 
        y_pos += 0    
    elif rank+minnodes == 10:    
        y_pos -= 0.0    
    elif rank+minnodes == 11:    
        y_pos -= 0.0   
    elif rank+minnodes == 12:    
        y_pos -= 0.0   
    elif rank+minnodes == 13:    
        y_pos -= 0.00
    elif rank+minnodes == 14:    
        y_pos -= 0.002    
    elif rank+minnodes == 15:    
        y_pos -= 0.004
    plt.text(1001.5, y_pos, str(rank+minnodes)+" nodes", fontsize=14, color=tableau20[rank])
plt.title('Back Propagation evaluated with Sigmoidal Function')
plt.savefig("sigBackprop.png", bbox_inches="tight")
plt.figure(2)
for rank, column in enumerate(datahb):    
    plt.plot(genrange,column,lw=2.5, color=tableau20[rank])
    y_pos = column[998]     
    if rank+minnodes == minnodes:    
       y_pos += 0    
    elif rank+minnodes == 4: 
        y_pos -= 0    
    elif rank+minnodes == 5:    
        y_pos += 0    
    elif rank+minnodes == 6:    
        y_pos -= 0    
    elif rank+minnodes == 7:    
        y_pos += 0    
    elif rank+minnodes == 8:    
        y_pos -= 0    
    elif rank+minnodes == 9: 
        y_pos += 0    
    elif rank+minnodes == 10:    
        y_pos -= 0.0    
    elif rank+minnodes == 11:    
        y_pos -= 0.0   
    elif rank+minnodes == 12:    
        y_pos -= 0.0   
    elif rank+minnodes == 13:    
        y_pos -= 0.00
    elif rank+minnodes == 14:    
        y_pos -= 0.002    
    elif rank+minnodes == 15:    
        y_pos -= 0.004
    plt.text(1001.5, y_pos, str(rank+minnodes)+" nodes", fontsize=14, color=tableau20[rank])
plt.title('Back Propagation evaluated with Step Edge Function')
plt.savefig("heaBackprop.png", bbox_inches="tight")
plt.figure(minnodes)
for rank, column in enumerate(datasa):    
    plt.plot(genrange,column,lw=2.5, color=tableau20[rank])
    y_pos = column[998]     
    if rank+minnodes == minnodes:    
       y_pos += 0    
    elif rank+minnodes == 4: 
        y_pos -= 0    
    elif rank+minnodes == 5:    
        y_pos += 0    
    elif rank+minnodes == 6:    
        y_pos -= 0    
    elif rank+minnodes == 7:    
        y_pos += 0    
    elif rank+minnodes == 8:    
        y_pos -= 0    
    elif rank+minnodes == 9: 
        y_pos += 0    
    elif rank+minnodes == 10:    
        y_pos -= 0.0    
    elif rank+minnodes == 11:    
        y_pos -= 0.0   
    elif rank+minnodes == 12:    
        y_pos -= 0.0   
    elif rank+minnodes == 13:    
        y_pos -= 0.00
    elif rank+minnodes == 14:    
        y_pos -= 0.002    
    elif rank+minnodes == 15:    
        y_pos -= 0.004
    plt.text(1001.5, y_pos, str(rank+minnodes)+" nodes", fontsize=14, color=tableau20[rank])
plt.title('Adaptive Back Propagation evaluated with Sigmoidal Function')
plt.savefig("sigAdaptive.png", bbox_inches="tight")
plt.figure(4)
for rank, column in enumerate(dataha):   
    plt.plot(genrange,column,lw=2.5, color=tableau20[rank])
    y_pos = column[998]     
    if rank+minnodes == minnodes:    
       y_pos += 0    
    elif rank+minnodes == 4: 
        y_pos -= 0    
    elif rank+minnodes == 5:    
        y_pos += 0    
    elif rank+minnodes == 6:    
        y_pos -= 0    
    elif rank+minnodes == 7:    
        y_pos += 0    
    elif rank+minnodes == 8:    
        y_pos -= 0    
    elif rank+minnodes == 9: 
        y_pos += 0    
    elif rank+minnodes == 10:    
        y_pos -= 0.0    
    elif rank+minnodes == 11:    
        y_pos -= 0.0   
    elif rank+minnodes == 12:    
        y_pos -= 0.0   
    elif rank+minnodes == 13:    
        y_pos -= 0.00
    elif rank+minnodes == 14:    
        y_pos -= 0.002    
    elif rank+minnodes == 15:    
        y_pos -= 0.004
    plt.text(1001.5, y_pos, str(rank+minnodes)+" nodes", fontsize=14, color=tableau20[rank])
plt.title('Adaptive Back Propagation evaluated with Step Edge Function')
plt.savefig("heaAdaptive.png", bbox_inches="tight")

plt.figure(5)


ind = np.arange(maxnodes+1-minnodes)  # the x locations for the groups
width = 0.25       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, mindatasb, width, color='r')

rects2 = ax.bar(ind + width, mindatasa, width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width)
ax.set_xticklabels(np.arange(maxnodes+1-minnodes)+3)

ax.legend((rects1[0], rects2[0]), ('Classic', 'Adaptive'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.savefig("sigminerror.png",bbox_inches="tight")