from matplotlib.pyplot import *

figure(figsize=(10, 5), dpi=3000)
plot([1,3,4,5], [3,3,3,3], color="r", label="DNS simulation")
hold("on")
errorbar([1], [3], yerr=[[2], [3]], fmt="o", label=r"Experiment mean $\pm 95\%$ C.I.")
#plot([1,3,4,5], [3,3,3,3], label="17M")
#plot([1,3,4,5], [3,3,3,3], label="28M")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
#show()
savefig("legend.png")
