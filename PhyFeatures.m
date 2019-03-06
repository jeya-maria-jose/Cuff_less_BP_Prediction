%The features are ai,lasi, S01,S02,S03,S04,Diff,Diff1,Difff
tic
clc;
%clear all;
FILE=[];
%close all; 
%load('Part_4');
for d=1:3000
    try
    d    
Y=(Part_4{1,d});  
O1P=Y(1,1:1000);
BP=Y(2,1:1000);
O1E=Y(3,1:1000); 
%plot(O1P);
Fs=125;
Ts=1/125; %sampling frequency=125Hz
T =(0:0.008:7.999); %time vector based on sampling rate
W1=0.5/62.5;
W22=5/62.5;
[b,a]=butter(3,[W1,W22]); % Bandpass digital filter design 
%h = fvtool(b,a); % Visualize filter
FP = filtfilt(b,a,O1P); 
%plot(FP)
[Fy]=gradient(FP);
[Fy1]=gradient(Fy);
%plot(Fy1);
%Finding Inflection points
for j=1:1000               
  if ( Fy1(j)< 0.0002 && Fy1(j)>0) || (Fy1(j)> -0.0002 && Fy1(j)<0) %second derivative close to zero as it is discrete time series
      Fy1(j)= 0;    
  end                      
end   
n=1;
i=[];
for j=1:1000               
  if (Fy1(j)== 0) 
   i(n)= j;
   n=n+1; %Counting number of inflection points
  end                      
end  
p1=1;p2=1;p3=1;p4=1;p5=1;p6=1;
%Separating the signal into 6 parts to find IPA - four area ratios
for k1=1:6:n-1
   h1(p1)=i(k1);    
   l1(p1)=O1P(h1(p1));
   p1=p1+1;
end  
for k2=2:6:n-1
   h2(p2)=i(k2);   
   l2(p2)=O1P(h2(p2));
   p2=p2+1;
end
for k3=3:n-1
   h3(p3)=i(k3);   
   l3(p3)=O1P(h3(p3));
   p3=p3+1;
end
for k4=4:6:n-1
   h4(p4)=i(k4);    
   l4(p4)=O1P(h4(p4));
   p4=p4+1;
end
for k5=5:6:n-1
   h5(p5)=i(k5);    
   l5(p5)=O1P(h5(p5));
   p5=p5+1;
end
for k6=6:6:n-1
   h6(p6)=i(k6);    
   l6(p6)=O1P(h6(p6));
   p6=p6+1;
end 
%AREA of the region between point (1,2) (1,3) (1,4) (1,5) (1,6)
i1=1;
for x1=1:fix((n-1)/6)  
    e1=1;
    for q1= h1(x1):h2(x1)
        a1(e1)= O1P(q1);
        e1=e1+1;
    end
    r1(i1)=sum(a1);
    i1=i1+1;
end
i2=1;
for x1=1:fix((n-1)/6)  
    e1=1;
    for q2=h1(x1):h3(x1)
        a2(e1)= O1P(q2);
        e1=e1+1;
    end
    r2(i2)=sum(a2);
    i2=i2+1;
end
i3=1;
for x1=1:fix((n-1)/6)
    e1=1;
    for q3=h1(x1):h4(x1)
        a3(e1)= O1P(q3);
        e1=e1+1;
    end
    r3(i3)=sum(a3);
    i3=i3+1;
end
i4=1;
for x1=1:fix((n-1)/6) 
    e1=1;
    for q4=h1(x1):h5(x1)
        a4(e1)= O1P(q4);
        e1=e1+1;
    end
    r4(i4)=sum(a4);
    i4=i4+1;
end
i5=1;
for x1=1:fix((n-1)/6)
    e1=1;
    for q5=h1(x1):h6(x1)
        a5(e1)= O1P(q5);
        e1=e1+1; 
    end
    r5(i5)=sum(a5);
    i5=i5+1;
end

for x2= 1:fix((n-1)/6)
    s1(x2)=(abs(r5(x2)-r1(x2)))/r1(x2);
    s2(x2)=(abs(r5(x2)-r2(x2)))/r2(x2);
    s3(x2)=(abs(r5(x2)-r3(x2)))/r3(x2);
    s4(x2)=(abs(r5(x2)-r4(x2)))/r4(x2);
end
S01=mean(s1);
S02=mean(s2);
S03=mean(s3);
S04=mean(s4); %S01,S02,S03,S04 are the required Area Ratios(IPA)- features.
ip=0;
%Now for Augmentation Index we need the ratio of amplitude between Systolic
%peak and the next inflection point.
[Peak,locp]=max(FP);
for u1=1:n-1
  if( i(u1)>locp) 
  ip=i(u1);
  break
  end
end  
%v1=min(FP);
%LASI is the inverse of time difference b/w systolic peak and the next
%inflection point
lasi=1/(T(ip)-T(locp));
%for o=1:1000
%FP(o)=FP(o)+abs(v1);
%end
v2=max(FP);
for o=1:1000
FP1(o)=FP(o)/v2;
end
%AI value = Ratio of amplitudes
ai=FP1(locp)/FP1(ip);
%Endofmyc---
N=length(O1P);
%plot(abs(fft(O1P)));
fbins=(0:N-1)*Fs/N;
%figure;
%plot(fbins,abs(fft(O1P)));
T1=movsum(Fy1,3);
%plot(T1);
W11=0.5/62.5;
W22=40/62.5;
[b,a]=butter(3,[W11,W22]); % Bandpass digital filter design 
FP1 = filtfilt(b,a,O1E); 
A=detrend(FP1);
D=movmax(T1,3); % second derivative of PPG signal
%plot(D);
% Adaptive threshold for peak detection
  wm=0.2;  % assume weights for updates
  wf=0.5;   % " " " " "
  THm=0.4*max(D); % assume threshold limits
  THf=0.8*max(D); % " " " " "
  [pk01, loc01]=findpeaks(D, 'MinpeakHeight', 0.8*max(D)); % calculating peaks for defined threshold limits
  [pk03, loc03]=findpeaks(D, 'MinpeakHeight', 0.4*max(D)); % " " " " "
  loopcounter=0;
  while(length(pk01)~=length(pk03))&& loopcounter<=10  % to update the weights and run until 
                                        %two different threshold equals the peak value count
      a=1;
       del=THm-THf;
       THmm=THm-wm*del;
       THff=THf-wf*del;
       [pk01,loc01]=findpeaks(D,'Minpeakheight',THmm);
       [pk03,loc03]=findpeaks(D,'MinpeakHeight',THff);
       loopcounter=loopcounter+1;
  end
h=(zeros(1,1000));
for j= 1:length(loc01)
   h(loc01(j))=1;
end
h(h==1)=pk01;
[C,Lag]=xcorr(A,h);
%plot(Lag/Fs, C);
[~, I]= max(abs(C));
Diff=Lag(I)/Fs;  % PATd
U=xcorr(FP1,Fy1);
%plot(U);

%PAT peak
[D L]=xcorr(A,O1P);
[~,I1]=max(abs(D));
Diff1=L(I1)/Fs;

% PAT foot
PPG1=max(O1P)-O1P;  % to find out min peak of PPG signal
[E L1]=xcorr(A,PPG1);
[~,I11]=max(abs(E));
Difff=L1(I11)/Fs;

filex=[d+9000 ai lasi S01 S02 S03 S04 Diff Diff1 Difff];
FILE=[FILE;filex];
    catch
        continue
    end    
end
toc

