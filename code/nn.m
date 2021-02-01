firstTrainingAudioFile   = "f.mp3";
secondTrainingAudioFile = "m.mp3";
C=1;%����ǿ����������
firstsongTrain   = C*audioread(firstTrainingAudioFile);
secondsongTrain = audioread(secondTrainingAudioFile);

 
L=500000;%����ֵ
firstsongTrain   = firstsongTrain(L:2*L);
secondsongTrain = secondsongTrain(L:2*L);%ѵ����
firstValidationAudioFile   ="f.mp3";
secondValidationAudioFile =  "m.mp3";

firstsongValidate   = C*audioread(firstValidationAudioFile);
secondsongValidate = audioread(secondValidationAudioFile);

 
 L1=1000000;
firstsongValidate   = firstsongValidate(3*L1:4*L1);
secondsongValidate = secondsongValidate(2.5*L1:3.5*L1);%��֤��

% ��ѵ���ź����ŵ���ͬ�Ĺ��ʡ�����֤�ź����ŵ���ͬ�Ĺ��ʡ�
firstsongTrain  =firstsongTrain/norm(firstsongTrain);%ѵ����,
secondsongTrain = secondsongTrain/norm(secondsongTrain);
firstsongValidate  = firstsongValidate/norm(firstsongValidate);%��֤��
secondsongValidate = secondsongValidate/norm(secondsongValidate);

mixTrain = firstsongTrain + secondsongTrain;
mixTrain = mixTrain / max(mixTrain);

mixValidate = firstsongValidate + secondsongValidate;
mixValidate = mixValidate / max(mixValidate);
WindowLength  = 128;
FFTLength     = 128;
OverlapLength = 128-1;
Fs            = 44000;
win           = hann(WindowLength,"periodic");
audiowrite('est_mix.wav',mixValidate,Fs);
P_mix0 = stft(mixTrain,'Window',win,'OverlapLength',OverlapLength,'FFTLength',FFTLength);
P_f    = abs(stft(firstsongTrain,'Window',win,'OverlapLength',OverlapLength,'FFTLength',FFTLength));
P_s    = abs(stft(secondsongTrain,'Window',win,'OverlapLength',OverlapLength,'FFTLength',FFTLength));
N      = 1 + FFTLength/2;
P_mix0 = P_mix0(N-1:end,:);%����ѵ�� STFT��
P_f    = P_f(N-1:end,:);
P_s    = P_s(N-1:end,:);
P_mix = log(abs(P_mix0) + eps);%�Ի��� STFT ȡ������ͨ����ֵ�ͱ�׼�����Щֵ���й�һ��
MP    = mean(P_mix(:));
SP    = std(P_mix(:));
P_mix = (P_mix - MP) / SP;
%������֤ STFT��
P_Val_mix0 = stft(mixValidate,'Window',win,'OverlapLength',OverlapLength,'FFTLength',FFTLength);
P_Val_f    = abs(stft(firstsongValidate,'Window',win,'OverlapLength',OverlapLength,'FFTLength',FFTLength));
P_Val_s    = abs(stft(secondsongValidate,'Window',win,'OverlapLength',OverlapLength,'FFTLength',FFTLength));

 P_Val_mix0 = P_Val_mix0(N-1:end,:);
P_Val_f    = P_Val_f(N-1:end,:);
P_Val_s    = P_Val_s(N-1:end,:);

P_Val_mix = log(abs(P_Val_mix0) + eps);%����֤ STFT ȡ������ͨ����ֵ�ͱ�׼�����Щֵ���й�һ��
MP        = mean(P_Val_mix(:));
SP        = std(P_Val_mix(:));
P_Val_mix = (P_Val_mix - MP) / SP;

maskTrain    = P_f ./ (P_f + P_s + eps);%ѵ�������Ϊ��һ�׸����������Ĥ
maskValidate = P_Val_f ./ (P_Val_f + P_Val_s + eps);%��֤�����Ϊ��һ�׸����������Ĥ


% ����Ԥ�������Ŀ���źŴ�����СΪ (65,20) �ķֿ顣Ϊ�˻�ø���ѵ���������������ֿ�֮��ʹ�� 10 ������Ϊ�ص�����
seqLen        = 20;
seqOverlap    = 10;
mixSequences  = zeros(1 + FFTLength/2,seqLen,1,0);
maskSequences = zeros(1 + FFTLength/2,seqLen,1,0);

loc = 1;
while loc < size(P_mix,2) - seqLen
    mixSequences(:,:,:,end+1)  = P_mix(:,loc:loc+seqLen-1); %ѵ������stft
    maskSequences(:,:,:,end+1) = maskTrain(:,loc:loc+seqLen-1); %ѵ���������Ե���������Ĥ

    loc                        = loc + seqOverlap;
end
mixValSequences  = zeros(1 + FFTLength/2,seqLen,1,0);
maskValSequences = zeros(1 + FFTLength/2,seqLen,1,0);
seqOverlap       = seqLen;

loc = 1;
while loc < size(P_Val_mix,2) - seqLen
    mixValSequences(:,:,:,end+1)  = P_Val_mix(:,loc:loc+seqLen-1); %��֤����stft
    maskValSequences(:,:,:,end+1) = maskValidate(:,loc:loc+seqLen-1); %��֤�������Ե���������Ĥ
    loc                           = loc + seqOverlap;
end
% ѵ���ź��ع�
mixSequencesT  = reshape(mixSequences,    [1 1 (1 + FFTLength/2) * seqLen size(mixSequences,4)]);
mixSequencesV  = reshape(mixValSequences, [1 1 (1 + FFTLength/2) * seqLen size(mixValSequences,4)]);
maskSequencesT = reshape(maskSequences,   [1 1 (1 + FFTLength/2) * seqLen size(maskSequences,4)]);
maskSequencesV = reshape(maskValSequences,[1 1 (1 + FFTLength/2) * seqLen size(maskValSequences,4)]);
numNodes = (1 + FFTLength/2) * seqLen;

layers = [ ...
    
    imageInputLayer([1 1 (1 + FFTLength/2)*seqLen],"Normalization","None")
    
    fullyConnectedLayer(numNodes)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(numNodes)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(numNodes)
    reluLayer
    dropoutLayer(0.1)
%  fullyConnectedLayer(numNodes)
%      reluLayer

    regressionLayer
    
    ];
maxEpochs     = 12;
miniBatchSize = 64;

options = trainingOptions("adam", ...
    "MaxEpochs",maxEpochs, ...
    "MiniBatchSize",miniBatchSize, ...
    "SequenceLength","longest", ...
    "Shuffle","every-epoch",...
    "Verbose",0, ...
    "Plots","training-progress",...
    "ValidationFrequency",floor(size(mixSequencesT,4)/miniBatchSize),...
    "ValidationData",{mixSequencesV,maskSequencesV},...
    "LearnRateSchedule","piecewise",...
    "LearnRateDropFactor",0.9, ...
    "LearnRateDropPeriod",1);
doTraining = true;
if doTraining
    CocktailPartyNet = trainNetwork(mixSequencesT,maskSequencesT,layers,options);
else
    s = load("CocktailPartyNet.mat");
    CocktailPartyNet = s.CocktailPartyNet;
end
% ����֤Ԥ��������ݸ����硣����ǹ��Ƶ���Ĥ��
estimatedMasks0 = predict(CocktailPartyNet,mixSequencesV);
estimatedMasks0 = estimatedMasks0.';
estimatedMasks0 = reshape(estimatedMasks0,1 + FFTLength/2,numel(estimatedMasks0)/(1 + FFTLength/2));

% ���Ƶ�һ�׸�͵ڶ��׸������Ĥ��ͨ��Ϊ����Ĥ������ֵ�����Ƶ�һ�׸�͵ڶ��׸�Ķ�Ԫ��Ĥ��
% ������Ĥ
SoftfirstMask   = estimatedMasks0; 
SoftsecondMask = 1 - SoftfirstMask;%�õ��ڶ��׸����Ĥ
P_Val_mix0 = P_Val_mix0(:,1:size(SoftfirstMask,2));
P_First = P_Val_mix0 .* SoftfirstMask;%Ԥ��ĵ�һ�׸��stft
P_First = [conj(P_First(end-1:-1:2,:)) ; P_First ];
%�õ�Ԥ��ĵ�һ�׸���Ƶ
firstsong_est_soft = istft(P_First, 'Window',win,'OverlapLength',OverlapLength,'FFTLength',FFTLength,'ConjugateSymmetric',true);
firstsong_est_soft = firstsong_est_soft / max(abs(firstsong_est_soft));

range = (numel(win):numel(firstsong_est_soft)-numel(win));
t     = range * (1/Fs);
audiowrite('est1soft.wav',firstsong_est_soft,Fs);
figure(1)%������Ĥ
subplot(2,1,1)
plot(t,firstsongValidate(range))
title("��ʼ��һ�׸���Ƶ")
xlabel("ʱ�� (s)")
grid on
subplot(2,1,2)
plot(t,firstsong_est_soft(range))
xlabel("ʱ�� (s)")
title("�����������ĵ�һ�׸���Ƶ(��Ĥ)")
grid on;


P_Second = P_Val_mix0 .* SoftsecondMask;

P_Second = [conj(P_Second(end-1:-1:2,:)) ; P_Second ];

secondsong_est_soft = istft(P_Second, 'Window',win,'OverlapLength',OverlapLength,'FFTLength',FFTLength,'ConjugateSymmetric',true);
secondsong_est_soft = secondsong_est_soft / max(secondsong_est_soft);
range = (numel(win):numel(firstsong_est_soft)-numel(win));
t     = range * (1/Fs);
audiowrite('est2soft.wav',secondsong_est_soft,Fs);
figure(2)
subplot(2,1,1)
plot(t,secondsongValidate(range))
title("��ʼ�ڶ��׸���Ƶ")
xlabel("ʱ�� (s)")
grid on
subplot(2,1,2)
plot(t,secondsong_est_soft(range))
xlabel("ʱ�� (s)")
title("�����������ĵڶ��׸���Ƶ(��Ĥ)")
grid on

% ������ԪĤ
HardFirstMask   = (SoftfirstMask >= 0.5);
HardSecondMask = (SoftfirstMask < 0.5);
P_First = P_Val_mix0 .* HardFirstMask;
P_First = [conj(P_First(end-1:-1:2,:)) ; P_First ];

first_est_hard = istft(P_First,'Window',win,'OverlapLength',OverlapLength,'FFTLength',FFTLength,'ConjugateSymmetric',true);
first_est_hard = first_est_hard / max(first_est_hard);
P_First = P_Val_mix0 .* HardFirstMask;

P_First = [conj(P_First(end-1:-1:2,:)) ; P_First ];

first_est_hard = istft(P_First,'Window',win,'OverlapLength',OverlapLength,'FFTLength',FFTLength,'ConjugateSymmetric',true);
first_est_hard = first_est_hard / max(first_est_hard);

P_Second = P_Val_mix0 .* HardSecondMask;

P_Second = [conj(P_Second(end-1:-1:2,:)) ; P_Second ];

second_est_hard = istft(P_Second, 'Window',win,'OverlapLength',OverlapLength,'FFTLength',FFTLength,'ConjugateSymmetric',true);
second_est_hard = second_est_hard / max(second_est_hard);
range = (numel(win):numel(firstsong_est_soft)-numel(win));
t   = range * (1/Fs);

figure(3)
subplot(2,1,1)
plot(t,firstsongValidate(range))
title("��ʼ��һ�׸���Ƶ")
xlabel("ʱ�� (s)")
grid on
subplot(2,1,2)
plot(t,first_est_hard(range))
xlabel("ʱ�� (s)")
title("�����������ĵ�һ�׸���Ƶ(��ԪĤ)")
grid on
audiowrite('est1binary.wav',first_est_hard,Fs);

figure(4)
subplot(2,1,1)
plot(t,secondsongValidate(range))
title("��ʼ�ڶ��׸���Ƶ")
xlabel("ʱ�� (s)")
grid on
subplot(2,1,2)
plot(t,second_est_hard(range))
title("�����������ĵڶ��׸���Ƶ(��ԪĤ)")
xlabel("ʱ�� (s)")
grid on

audiowrite('est2binary.wav',second_est_hard,Fs);
figure(6)
mixValidate=mixValidate(1:size(t,2));
plot(t,mixValidate);
title("�����Ƶ")
xlabel("ʱ�� (s)")
