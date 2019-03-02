import pandas as pd
import numpy as np
import csv

sbp = list()
dbp = list()
real_BP = list()
with open('/home/jeyamariajose/Projects/fyp/bp.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter = ',')
    print csv_reader
    for row in csv_reader:
        #ptt.append(float(row[2]))
        sbp.append(float(row[0]))
        dbp.append(float(row[1]))

    real_BP = list()
    for i in range(len(sbp)):
        BP_actual = (2*dbp[i] + sbp[i])/3
        real_BP.append(BP_actual)

real_BP = np.transpose(real_BP)
with open('mbp.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for word in real_BP:
        employee_writer.writerow([word])
