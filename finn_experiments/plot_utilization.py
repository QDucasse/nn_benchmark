
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glob import glob
import json

# Get paths to json files
results_path = ""

tested_networks = list(sorted(glob(results_path+"QuantTFC_A*")))
ppr_json_paths = [glob(t_n+"/*.json")[0] for t_n in tested_networks]
print(ppr_json_paths)

# Read json data

result_dict = {}

for i, ppr_json in enumerate(ppr_json_paths):
    with open(ppr_json, 'r') as f:
        res = json.load(f)
    # Only extract the percentages and reformat as dictionary
    for key in res.keys():
        data_dict = {}
        for data_point in res[key]:
            # skip empty data points
            if len(data_point) == 0:
                continue
            data_dict[data_point[0]] = data_point[-1]
        res[key] = data_dict

    result_dict[tested_networks[i]] = res

keys_of_interest = ['1. Slice Logic', '3. Memory']

plotting_dataframes = {}
for key in keys_of_interest:
    plotting_dataframes[key] = pd.DataFrame(columns=list(result_dict[tested_networks[0]][key].keys()))
    for net in tested_networks:
        res = pd.DataFrame([list(result_dict[net][key].values())], columns=list(result_dict[net][key].keys()), index=[net])
        plotting_dataframes[key] = plotting_dataframes[key].append(res)

# Plot data

for key in keys_of_intrest:
    plotting_dataframes[key].T.plot.bar(rot=45)
    plt.ylabel("Utilization [%]")
    plt.tight_layout()
    plt.savefig(key+".png", dpi=400)
    plt.show()
