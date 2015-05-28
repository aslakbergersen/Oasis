from numpy import *
from os import path, listdir

data_path = path.join("..", "nozzle_results", "data")
for folder in ["121", "126", "127"]:
    for stat in listdir(path.join(data_path, folder, "Stats")):
        if stat != "Points":
            tmp_arr = load(path.join(data_path, folder, "Stats", stat))
            num = float(stat.split("_")[-1])
            tmp_arr = tmp_arr / num
            tmp_arr.dump(path.join(data_path, folder, "Stats", stat))
