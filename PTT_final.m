tic
clc;
clear all;
close all; 
load('Part_4');
FILE = []
for d=1:3000
    d
    
Y=(Part_4{1,d});  
O1P=Y(1,1:1000);
BP=Y(2,1:1000);
O1E=Y(3,1:1000); 
Fs=125;
Ts=1/125; %sampling frequency=125Hz
T =(0:0.008:7.999); %time vector based on sampling rate
W1=0.5/62.5;
W2=5/62.5;
[b,a]=butter(3,[W1,W2]); % Bandpass digital filter design 
%h = fvtool(b,a); % Visualize filter
FP = filtfilt(b,a,O1P); 
[Fy]=gradient(FP);
for j=1:1000
  if Fy(j)<= 0
      Fy(j) = 0;
  end
end
T1=movsum(Fy,3);
plot(T);
W1=0.5/62.5;
W2=40/62.5;
[b,a]=butter(3,[W1,W2]); % Bandpass digital filter design 
FP1 = filtfilt(b,a,O1E); 
A=detrend(FP1);
E=detrend(FP);
D=movmax(T1,3);
[pk1, loc1]=findpeaks(D);
findpeaks(D);
h=(zeros(1,1000));
for i= 1:length(loc1)
   h(loc1(i))=1;
end
h(h==1)=pk1;
[C,Lag]=xcorr(A,h);
%plot(Lag/Fs, C);
[~, I]= max(abs(C));
Diff=Lag(I)/Fs;
filerow= [real(abs(Diff))];
FILE = [FILE;filerow];
end
csvwrite('ptt_newpart4.csv',FILE);
