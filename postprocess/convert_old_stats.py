from numpy import *
from os import path, listdir

data_path = path.join("..", "nozzle_results", "data")
for folder in ["25"]:#listdir(data_path):
    for stat in listdir(path.join(data_path, folder, "Stats")):
        if stat != "Points":
            tmp_arr = load(path.join(data_path, folder, "Stats", stat))
            num = float(stat.split("_")[-1])
            tmp_arr = tmp_arr *0.8 #/ num
            tmp_arr.dump(path.join(data_path, folder, "Stats", stat))
