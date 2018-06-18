#!/usr/bin/python
import xml.etree.ElementTree as ET
import sys
import numpy as np

def main():
    xml_path = sys.argv[1]
    csv_path = sys.argv[2]
    tree = ET.parse(xml_path)
    root = tree.getroot()
    seg = np.zeros((600, 600))
    for i_row in range(600):
        for i_column in range(600):
            seg[i_row, i_column] = float(root[5][i_row][i_column][0].text)
    np.savetxt(csv_path, np.transpose(np.abs(seg)))

if __name__ == "__main__":
    main()
