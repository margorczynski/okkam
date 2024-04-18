#!/bin/python

import csv
import math

def test_dataset_function(x, y, z):
    return (x**2 + x * y * z + math.sqrt(y) * math.log(z) - z**3) * math.sqrt(x + y + z)

if __name__ == "__main__":
    with open('test_dataset.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(1, 101):
            x, y, z = i, i, i
            result = test_dataset_function(x, y, z)
            writer.writerow([x, y, z, result])