% 
 clc;
%  clear; 
 close all;

          
Im = double(imread('Img1.png'));
  
[n1,n2,n3] = size(Im); 
maxP = max(Im(:));
QIm=quaternion(Im(:,:,1),Im(:,:,2),Im(:,:,3)); % Original
sizeData = size(QIm);
Ob_QIm=vector(zerosq(n1,n2));
  SR=0.25;  %sampling ratio 
Omega = find(rand(n1,n2)<=SR); %Random sampling


Nway = size(QIm);
data=QIm(Omega);
Ob_QIm(Omega)=data; % Observe

q2double = @(X) double2q(X,'inverse');
psnr1 = psnr(Im./255, q2double(Ob_QIm)./255);
ssim1 = ssim(Im./255, q2double(Ob_QIm)./255);


   r1 = 150;   
   beta =0.001;
%    gama =  20;
gama = 30;

        opts.XTrue = Im;
        opts.r_1 = r1;
        opts.beta = beta;
        opts.gama = gama;
        opts.tol = 1e-4;
        opts.maxIter = 100;

         %% initialize D
         
         [U1, ~, ~]   = qsvd(Ob_QIm);
         U1    = quaternion(U1(:,:,1),U1(:,:,2),U1(:,:,3),U1(:,:,4));
          
         DD = U1(:,1:r1);
         D{1} =DD' ;
          
         [U2, ~, ~]   = qsvd(Ob_QIm);
         U2    = quaternion(U2(:,:,1),U2(:,:,2),U2(:,:,3),U2(:,:,4));
         
         DD = U2(:,1:r1);
         D{2} = DD';

         opts.XTrue = Im;
         figure;
[TX,errList,PSNR_best,SSIM_best] = CLQNMF_PnP(Ob_QIm, Omega, D,opts);
% [TX,errList,PSNR_best,SSIM_best] = CLQNMF(Ob_QIm, Omega, D,opts);

         

   
fprintf('%4.2f/%6.4f\n',PSNR_best, SSIM_best);
% end

