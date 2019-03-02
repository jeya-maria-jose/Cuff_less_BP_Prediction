tic
clc;
clear all;
close all; 
load('Part_1');
FILE = []
filename=""
for d=1:3

Y=(Part_1{1,d});  
O1P=Y(1,1:1000);
BP=Y(2,1:1000);
O1E=Y(3,1:1000); 

O1P = transpose(O1P);

O1E = transpose(O1E);

filerow= [O1P O1E];
%FILE = [FILE;filerow];

filename = "check"+num2str(d)+".csv";
filename = char(filename);

csvwrite(filename,filerow);

end


toc
    