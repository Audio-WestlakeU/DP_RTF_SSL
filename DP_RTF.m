function dprtf = DP_RTF(x,xfs,t60)

% This method estimates the direct-path relative transfer function for the
% single static speaker case in a batch way. 
%
%% Input
% x : binaural audio signal, with the size of No. of samples x 2
% fs: sampling rate of x
% t60: reverberation time in ms. An approximate t60 should be given to determine the CTF length, otherwise using the defult: 400 ms 
%
%% Output
% dprtf: No. of frequency x 1, the estimation of the dprtf b_{0,k}/a_{0,k}.  

% Author: Xiaofei Li, INRIA Grenoble Rhone-Alpes
% Copyright: Perception Team, INRIA Grenoble Rhone-Alpes
% The algorithm is described in the paper:  
% Xiaofei Li, Laurent Girin, Radu Horaud, Sharon Gannot. Estimation of the Direct-Path Relative Transfer Function for Supervised Sound-Source Localization. IEEE/ACM Transactions on Audio, Speech and Language Processing, Institute of Electrical and Electronics Engineers, 2016, 24 (11), pp.2171 - 2186.

if nargin<3
     t60 = 400;
end 

% STFT setup
fs = 16000;               % target sampling rate
nfft = 256;               % window length of STFT, 16 ms
ninc = 0.25*nfft;         % step size of STFT, 4 ms. Alternatively, it can be tuned to 8 ms

M = size(x,2);            % Microphone number, actually equals 2

% parameters
D = 30;                   % PSD smoothing length, corresponds to 30*4=120 s
ctf_length = t60/5;       % CTF length in ms
Q = min(50,ceil(ctf_length*fs/(1000*ninc))); % CTF length in frames

% STFT
X = stft(resample(x(:,1),fs,xfs),nfft,ninc,hamming(nfft));
X(:,:,2) = stft(resample(x(:,2),fs,xfs),nfft,ninc,hamming(nfft));
[K,P,~] = size(X);        % NO. of frequencies and frames

% Construct the convolution vector of microphone signals for all frames and frequencies 
XQ = zeros(K,P-Q+1,M,Q);
for q = 0:Q-1
    XQ(:,:,:,q+1) = X(:,Q-q:P-q,:);
end

% XQ multiply y(p) 
XQy = bsxfun(@times,XQ,conj(XQ(:,:,2,1)));

% Auto- and cross-PSDs between [x y] and y(p) 
PSD = zeros(K,P-Q-D+2,M,Q);
for d = 0:D-1
    PSD = PSD+XQy(:,D-d:P-Q+1-d,:,:);
end
PSD = PSD/D;
Ps = size(PSD,2);

% Minimum controled maximum thresholds
[r1,r2] = MCMT(D,1,Ps);   

% Speech and noise frames classification
psd = PSD(:,:,2,1);
minpsd = min(psd,[],2);
l1 = psd>r1*repmat(minpsd,[1,Ps]);    % speech class
l2 = psd<r2*repmat(minpsd,[1,Ps]);    % noise class

% Frequency-wise dprtf estimation 
dprtf = zeros(K,1);
for k = 1:K    
    
    % Indices of segments
    kl1 = find(l1(k,:));    
    if length(kl1)<2*Q-1
        continue;
    end    
    kl2 = find(l2(k,:));   
    
    % Nearest noise segments 
    dis = abs(bsxfun(@minus,kl1,kl2'));
    [~,kl12_ind] = min(dis,[],1);
    kl12 = kl2(kl12_ind);
    
    % Spectral subtraction
    PSDss = squeeze(PSD(k,kl1,:,:) - PSD(k,kl12,:,:));
      %PSDss = squeeze(PSD(k,kl1,:,:)); % no spectral subtraction
    
    % dprtf estimate
    Phizy = [squeeze(PSDss(:,1,:)),squeeze(PSDss(:,2,2:end))];
    phiyy = PSDss(:,2,1);
    gk = Phizy\phiyy;
    dprtf(k) = gk(1);   
end


