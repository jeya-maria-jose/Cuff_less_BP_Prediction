tic
clc;
%clear all;
close all; 
load('Part_4');
FILE=[];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for d=1:3000
    d

Y=(Part_4{1,d});
O1P=Y(1,1:1000);
BP=Y(2,1:1000);
O1E=Y(3,1:1000);
%{
figure('Name','ECG');
plot(O1E);
figure('Name','PPG');
plot(O1P);
%}
[Fy]=gradient(O1P);
%figure('Name','PPG 1st derivative');

%plot(Fy);
[Fy1]=gradient(Fy);  % 2nd derivative
%figure('Name', 'PPG 2nd derivative');

%plot(Fy1);
F=cat(1,O1P,Fy1,O1E);
%figure('Name', 'All concat');
%plot(F)
L=length(F);
Ts=1/125; %sampling frequency=125Hz
%t=linspace(0,10,0.008);
T =(0:0.008:7.999); %time vector based on sampling rate

[pk,loc]= findpeaks(O1P); % max value of PPG signal
PPG1=max(O1P)-O1P; % To find out the min peak of PPG
[pk1,loc1]=findpeaks(PPG1,'MinPeakHeight',0.0); % min value of PPG signal

sys_time=0;
for i=1:1:5
    sys_time = sys_time + T(loc(1,i))-T(loc1(1,i));
end
sys_time = sys_time/5;

dias_time=0;
for i=1:1:5
    dias_time = dias_time + T(loc1(1,i+1))-T(loc(1,i));
end


dias_time = dias_time/5;

v = [0.1,0.25,0.33,0.5,0.66,0.75];

ppg_21_st = [];
ppg_21_dt = [];
for j=1:1:6
for i=loc1(1,1):1:loc(1,1)
    if(O1P(1,i)>=(v(j)*pk(1,1)+pk1(1,1)))
        a=i;
   
        break
    end
end

for i=loc(1,1):1:loc1(1,2)
  
    if(O1P(1,i)<=(v(j)*pk(1,1)+ pk1(1,1)))
        b=i;
        
        break
    end
end

ppg_21_st(j) = (loc(1,1)-a)*0.008;
ppg_21_dt(j) = (b-loc(1,1))*0.008;

end


        
%findpeaks(PPG1,'MinPeakHeight',0.6);  % noise threshold
%figure('Name','min PPG after threshold');

%plot(PPG1);

% original [pk2,loc2]= findpeaks(O1E,'MinpeakHeight',0.6); % max value of ECG signal

[pk2,loc2]= findpeaks(O1E,'MinPeakHeight',0.6); % max value of ECG signal
%findpeaks(O1E,'MinPeakHeight',0.6);
% original [pk3,loc3]=findpeaks(Fy1,'MinpeakHeight',0.0398); % max value of DPPG signal
[pk3,loc3]=findpeaks(Fy1,'MinPeakHeight',0.003); % max value of DPPG signal


[m,n] = size (loc2); % to find out vector dimensions of ECG signal
[x,y] = size (loc3);

P1=T(loc2);
P=T(loc3);
P11=P1(1,1:n);
P2= P(1,1:y);
ptt=0;

temp=min(y,n);
range=min(temp,5);
for i=1:1:(range)
    
    ptt = ptt + abs(P2(1,(i))-P11(1,i));
    
    %PTT1(i) = P2(1,i)-P11(1,i)  % To find out the transit time btwn ECG and PPG signal
end
ptt = ptt/(range);

[lr,lr1] = size(loc1);
rationum=0;
ratioden=0;
ih=0;
il=0;
for i=1:1:lr1-1
    rationum = rationum + pk(1,i);
    ratioden = ratioden + pk1(1,i);
end
    %figure;
    
ih = rationum/(lr1-1);
il = ratioden/(lr1-1);

PIR=ih/il;

RR=diff(P1);  % to find time taken for 1 heartbeat
HR = 60./RR;
hrfinal=0;
[lr,lr1] = size (HR);
tlr1 = lr1;
for i=1:1:lr1
    t = HR(1,i);
    if t<=30||t>=200
        tlr1 = tlr1-1;
    else
        
        
        hrfinal = hrfinal + HR(1,i);
    end
end
hrfinal = hrfinal/(tlr1);

%figure
%subplot(3,1,1)
%plot(T,O1P);
%subplot(3,1,2)
%plot(T,O1E);
%subplot(3,1,3)
%plot(T,Fy1);
Yy = fft(O1P);
% % P2 = abs(Y/L);
%figure;plot(Yy);
Z=Yy(1);
Yy(1)=0;
S=real(ifft(Yy));

%figure;plot(S);
[pk4,loc4]=findpeaks(S); % max AC value of PPG signal
%xlabel('Time(s)');
%ylabel('Ac amplitude(V)');
[pk5,loc5]=findpeaks(BP); 

[lr,lr1] = size (loc4);
iftmax=0;
for i=1:1:lr1-1
    
    iftmax = iftmax + pk4(1,i);
end

meu = iftmax/(lr1-1);
    %figure;


alpha = il*sqrt(1060*hrfinal/meu);


%findpeaks(BP);
BP1=max(BP)-BP; % To find out the min peak of BP
[pk6,loc6]=findpeaks(BP1); % min value of BP(diastole) signal    
%findpeaks(BP1);


[lr,lr1] = size (loc5);
bpmax=0;
for i=1:1:lr1-1
    
    bpmax = bpmax + pk5(1,i);
end

bpmax = bpmax/(lr1-1);

[lr,lr1] = size (loc6);
bpmin=0;
for i=1:1:lr1-1
    
    bpmin = bpmin + pk6(1,i);
end

bpmin = bpmin/(lr1-1);


filerow1= [ppg_21_dt(1) ppg_21_st(1)+ppg_21_dt(1) ppg_21_dt(1)/ppg_21_st(1) ppg_21_dt(2) ppg_21_st(2)+ppg_21_dt(2) ppg_21_dt(2)/ppg_21_st(2) ppg_21_dt(3) ppg_21_st(3)+ppg_21_dt(3) ppg_21_dt(3)/ppg_21_st(3) ppg_21_dt(4) ppg_21_st(4)+ppg_21_dt(4) ppg_21_dt(4)/ppg_21_st(4) ppg_21_dt(5) ppg_21_st(5)+ppg_21_dt(5) ppg_21_dt(5)/ppg_21_st(5) ppg_21_dt(6) ppg_21_st(6)+ppg_21_dt(6) ppg_21_dt(6)/ppg_21_st(6) sys_time dias_time];
filerow= [real(alpha) real(PIR) real(ptt) real(bpmax) real(bpmin) hrfinal ih il meu];
 
%%%change here to include filerow
FILE = [FILE;filerow1];
 
end
csvwrite('ppg4.csv',FILE);
toc
