function [Imgbest,errList,PSNR_best,SSIM_best] = CLQNMF_PnP( T, Omega,D, opts)   
q2double = @(X) double2q(X,'inverse');
maxIter = opts.maxIter;
max_beta = 1e20;
r1 = opts.r_1;
beta = opts.beta;
gama = opts.gama;
[ncr(1,1), ncr(1,2)] = size(T);    
X  = T;
M  = X;

S = q2double(T);
eta2 = zeros(ncr(1,1), ncr(1,2),3); 

alpha =1;
% alpha =opts.alpha;

X(Omega) = T(Omega);
errList = zeros(maxIter, 1);   
D1 = D{1}; D2 = D{2};

Lambda = quaternion(zeros(ncr(1, 1), ncr(1, 2)));
Lambda2 = quaternion(zeros(ncr(1, 1), ncr(1, 2)));
PSNR_best = 0;
SSIM_best = 0;



for Iter = 1: maxIter    
   Xk = X;
   %% update Z
   Temp = D1 * (X + Lambda./beta) * D2';
   
   [U, SigmaY, V]   = qsvd(Temp);
   U    = quaternion(U(:,:,1),U(:,:,2),U(:,:,3),U(:,:,4));
   V    = quaternion(V(:,:,1),V(:,:,2),V(:,:,3),V(:,:,4));

   [m,n] = size(Temp);
   C = 5000;
   [tempDiagS,svp] = ClosedQNMF(SigmaY, alpha, C/beta);
   Z = U(:,1:svp)*diag(tempDiagS)*V(:,1:svp)'; 



   


    %% Update X
    EM = double2q( S - eta2/beta );
    X= (D1' * Z * D2 - Lambda/beta + EM + M - Lambda2/beta)/3;
    Xs=scalar(X);%取四元数矩阵X的实部
    X=vector(X);%只要虚部，实部为0
    Xa=beta*D1' * Z * D2 - Lambda + beta * EM + beta * M - Lambda2;
    Xa=vector(Xa);
    X(Omega)=(T(Omega)+Xa(Omega))/(1+3*beta);
    X=Xs+X;  
    X=vector(X);
   
    %% Update D1 
    Temp1 = beta *  (X + Lambda/beta) * D2' * Z';
    [UU, ~, VV]   = qsvd(Temp1);
    UU    = quaternion(UU(:,:,1),UU(:,:,2),UU(:,:,3),UU(:,:,4));
    VV    = quaternion(VV(:,:,1),VV(:,:,2),VV(:,:,3),VV(:,:,4));
    D1 = VV * UU(:,1:r1)';
  
    %% Update D2
    Temp2 =  beta * (X + Lambda/beta)' * D1' * Z;
    [UU, ~, VV]   = qsvd(Temp2);
    UU    = quaternion(UU(:,:,1),UU(:,:,2),UU(:,:,3),UU(:,:,4));
    VV    = quaternion(VV(:,:,1),VV(:,:,2),VV(:,:,3),VV(:,:,4));
    D2 = VV * UU(:,1:r1)';

    %% Update S
    TXX   = q2double(X);
    tempV = TXX + eta2./beta;
   S     = FFDNet(tempV./255,sqrt(gama./beta)./255).*255;



    %% Update M
    tempA = X + Lambda2/beta;
    M = PIB(tempA);

    
    Lambda = Lambda + beta * (X - D1' * Z * D2);
    eta2   = eta2 + beta * (TXX - S);
    Lambda2 = Lambda2 + beta * (X - M);
    
    
    X1 = max(part(X,2), 0);  X2 = max(part(X,3), 0);  X3 = max(part(X,4), 0); 
    X1 = min(X1, 255); X2 = min(X2, 255); X3 = min(X3, 255);
    TX(:,:,1)= X1 ;TX(:,:,2)= X2;TX(:,:,3)= X3;
    sXX = TX;
    GT = opts.XTrue;
    QGT = double2q(GT);
    QsXX = double2q(sXX);
    QsXX(Omega) = QGT(Omega);
    sXX = q2double(QsXX);
    imshow(uint8(sXX));
    stopC = (norm(Xk(:) -X(:)))/  norm(Xk(:));
    errList(Iter) = stopC; 
    
    PSNR=psnr(sXX./255,GT./255); 
    SSIM=ssim(sXX./255,GT./255);
    if PSNR > PSNR_best
        PSNR_best = PSNR;
        Imgbest = sXX;
    end
    if SSIM > SSIM_best
        SSIM_best = SSIM;
    end    

    if mod(Iter, 5) == 0      
           fprintf(' %4.2f/%6.4f      %4.2f/%6.4f\n' ,PSNR_best ,SSIM_best ,PSNR ,SSIM);    
        
         
    end

    if stopC < opts.tol
        break;
    else
        beta = min(beta * 1.1, max_beta);
    end

    


end

 