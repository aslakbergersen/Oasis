from matplotlib.pyplot import *


filenames = ["logs/Uz_0", "logs/k_0", "logs/epsilon_0", "logs/p_0", "logs/Ux_0", \
            "logs/Uy_0"]

for filename in filenames:
    file = open(filename, "r")

    R = []
    I = []

    for line in file.readlines():
        res, iter = line.split()
        R.append(float(res))
        I.append(float(iter))

    semilogy(R, I, legend=filename.split("/")[1], xlabel="Time",
            ylabel="Residual", title="Intial residual for all parameters")
    hold('on')
savefig("residual.png")
raw_input()
