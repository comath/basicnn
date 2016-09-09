import pandas
import matplotlib.pyplot as plt 

lables = []


heavyAD = pandas.read_csv("imgfilesBackprop/heaerrordataepochtot.csv")



tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
			(44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
			(148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
			(227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
			(188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 

# You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
# exception because of the number of lines being plotted on it.    
# Common sizes: (10, 7.5) and (12, 9)    
plt.figure(figsize=(12, 14))    
  
# Remove the plot frame lines. They are unnecessary chartjunk.    
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)    
  
# Ensure that the axis ticks only show up on the bottom and left of the plot.    
# Ticks on the right and top of the plot are generally unnecessary chartjunk.    
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()    
  
# Limit the range of the plot to only where the data is.    
# Avoid unnecessary whitespace.    
plt.ylim(0, 0.25)    
plt.xlim(0, 500)    
  
# Make sure your axis ticks are large enough to be easily read.    
# You don't want your viewers squinting to read your plot.    

  
# Provide tick lines across the plot to help your viewers trace along    
# the axis ticks. Make sure that the lines are light and small so they    
# don't obscure the primary data lines.    
for y in range(10, 91, 10):    
    plt.plot(range(0, 500), [y] * len(range(0, 500)), "--", lw=0.5, color="black", alpha=0.3)    
  
# Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")    
  
# Now that the plot is prepared, it's time to actually plot the data!    
# Note that I plotted the majors in order of the highest % in the final year.    

heavyAD.index = heavyAD['Gen']

numgen = range(0,500)
heavyAD = heavyAD.T

labels = heavyAD.columns




for rank, column in enumerate(labels):    
    # Plot each line separately with its own color, using the Tableau 20    
    # color set in order.

    plt.plot(numgen,    
            heavyAD[column].values[1:501],    
            lw=2.5, color=tableau20[rank]) 

plt.savefig("heavyBackprop.png", bbox_inches="tight")