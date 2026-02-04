function [X] = PIB(Z)
q2double = @(X) double2q(X,'inverse');
Y = q2double(Z);
Y1 = Y(:,:,1);
Y2 = Y(:,:,2);
Y3 = Y(:,:,3);
Y1(Y1>=255) = 255;
Y2(Y2>=255) = 255;
Y3(Y3>=255) = 255;
Y1(Y1<=0) = 0;
Y2(Y2<=0) = 0;
Y3(Y3<=0) = 0;
Y1 = round(Y1,0);
Y2 = round(Y2,0);
Y3 = round(Y3,0);
X(:,:,1) = Y1;
X(:,:,2) = Y2;
X(:,:,3) = Y3;
X = double2q(X);