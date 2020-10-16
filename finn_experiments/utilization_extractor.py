# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# This script is the courtesy of Hendrik Borras, https://github.com/HenniOVP

import json
import xml.etree.ElementTree as ET
import pandas as pd

lst_act = [2,3,4,5,6,7,8]
bitwidths_list = ["A{}W{}I{}".format(act,act,8) for act in lst_act]

if __name__ == "__main__":
    for bitwidths in bitwidths_list:
        # Constant parameters
        build_dir = "/home/quentin/Desktop/GitProjects/AI/finn/onnx_experiments/QuantTFC_" + bitwidths + "/"
        vivado_proj = build_dir + "vivado_zynq_proj_" + bitwidths + "/"
        report_path = "finn_zynq_link.runs/impl_1/top_wrapper_utilization_placed.rpt"

        # Read post place utilization report and save results

        with open(vivado_proj + report_path) as f:
            content = f.readlines()


        top_levels_to_parse = ["1. Slice Logic", "2. Slice Logic Distribution", "3. Memory", "4. DSP", "6. Clocking"]

        utilization_data = {}
        top_level_key = ""
        parse_enable = False
        waiting_for_table_start = False

        for line in content:
            # check if we stumbled upon one of the top level indicators
            if any(top in line for top in top_levels_to_parse):
                waiting_for_table_start = True
                # Find out which of them it was
                for i in range(len(top_levels_to_parse)):
                    if top_levels_to_parse[i] in line:
                        top_level_key = top_levels_to_parse[i]
                        break
                utilization_data[top_level_key] = []

            # Check for table start indicator
            if waiting_for_table_start:
                if "Available" in line:
                    parse_enable = True
                    waiting_for_table_start = False
                    continue

            # parse a line
            if parse_enable:
                # reached a table border
                if "+--" in line:
                    continue
                # reached end of table
                if "\n" == line or "* Note: Each Block RAM Tile only h" in line:
                    parse_enable = False
                    continue
                # parse table row
                line = line.strip()
                split_line = line.split("|")[1:-1]
                row_data = []
                for data_snipplet in split_line:
                    d = data_snipplet.strip()
                    try:
                        d = float(d)
                    except ValueError:
                        pass
                    row_data.append(d)
                # skip rows with emtpy elements
                if '' in row_data:
                    continue
                utilization_data[top_level_key].append(row_data)

        # Save as JSON
        # Save results to disk
        with open(build_dir+"/pynq_place_report_"+ bitwidths + ".json", 'w') as f:
            json.dump(utilization_data, f)

        utilization_data


        ## Read the synthesis report and save the results for later processing

        root = ET.parse(vivado_proj+"/synth_report.xml").getroot()

        # Read the table header of the synthesis report
        table_header = []
        for child in root[0][0][0]:
            attribs = child.attrib
            table_header.append(attribs['contents'])

        # Read the rest of the table
        table_cols = []
        for child1 in root[0][0][1:]:
            one_col = []
            for child2 in child1:
                content = child2.attrib['contents'].strip()
                # Everything that looks like an int should become an int
                try:
                    content = int(content)
                except ValueError:
                    pass
                one_col.append(content)
            table_cols.append(one_col)


        # Store data in a nicer format
        report_data = pd.DataFrame(table_cols, columns=table_header)

        # Save data to disk
        report_data.to_csv(build_dir+"/pynq_synthesis_report_"     + bitwidths + ".csv")

        report_data
