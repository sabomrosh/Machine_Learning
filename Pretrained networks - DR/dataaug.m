clear all;close all;clc;
d=dir('*.jpeg');
% for i=1:length(f)
%     a=find(f==0); 
%     t=find(f~=0);
%     %b=q(a,1);
%    % t=find(~ismember(q(:,1),a));
% end
    %d(b).name;
    v = randperm(length(d));
    q=fix(length(d)*.75);
    q = randperm(q);
    w1=v(1:3727);
    w2=v(3728:7454);
    for i=1:length(w1) 
       fname1=d(w1(i)).name;
       % N=imread([StimuliFolder files{v(i)} '.jpg'])
       TIF=imread(fname1);
       TIF = TIF(:,end:-1:1,:); 
%       %Converts to JPEG and gives it the .jpg extension
    %fname1 = [fname1(1:end-4),'.jpg'];
       Resultados='E:\sabooranew\kaggledata\data\train\patient\h';
      baseFileName = sprintf('%dh.jpeg', i); % e.g. "1.png"
       fullFileName = fullfile(Resultados, baseFileName);
       imwrite(TIF,fullFileName,'jpeg');
    end
    for i=1:length(w2) 
       fname2=d(w2(i)).name;
       % N=imread([StimuliFolder files{v(i)} '.jpg'])
       TIF=imread(fname2);
       TIF = TIF(end:-1:1,:,:); 
%       %Converts to JPEG and gives it the .jpg extension
    %fname1 = [fname1(1:end-4),'.jpg'];
       Resultados='E:\sabooranew\kaggledata\data\train\patient\v';
      baseFileName = sprintf('%dv.jpeg', i); % e.g. "1.png"
       fullFileName = fullfile(Resultados, baseFileName);
       imwrite(TIF,fullFileName,'jpeg');
    end
       for i=1:length(q) 
       fname3=d(q(i)).name;
       % N=imread([StimuliFolder files{v(i)} '.jpg'])
       TIF=imread(fname3);
       TIF = TIF(end:-1:1,end:-1:1,:); 
%       %Converts to JPEG and gives it the .jpg extension
    %fname1 = [fname1(1:end-4),'.jpg'];
       Resultados='E:\sabooranew\kaggledata\data\train\patient\vh';
      baseFileName = sprintf('%dvh.jpeg', i); % e.g. "1.png"
       fullFileName = fullfile(Resultados, baseFileName);
       imwrite(TIF,fullFileName,'jpeg');
    end