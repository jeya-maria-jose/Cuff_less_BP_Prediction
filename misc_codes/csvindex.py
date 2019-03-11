#f=open('cleaned_dataset_with_physio.csv')
f1=open('bp_test_new.csv')

fw=open('class_bp_test.csv','w')

mbp = list()
import csv
with open('bp_test_new.csv', 'r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter = ',')
	print csv_reader
	for column in csv_reader:
		
		mbp.append(column[2])

cl_list = list()

for idx in mbp:
	
	

	if float(float(idx))>60 and float(idx)<=70:
		cl = 0
	if float(idx)>70 and float(idx)<=80:
		cl = 1
	if float(idx)>80 and float(idx)<=90:
		cl = 2
	if float(idx)>90 and float(idx)<=100:
		cl = 3
	if float(idx)>100 and float(idx)<=110:
		cl = 4
	if float(idx)>110 and float(idx)<=120:
		cl = 5
	if float(idx)>120 and float(idx)<=130:
		cl = 6
	
	fw.write(str(cl)+"\n")


# f.close()
f1.close()
fw.close()

	