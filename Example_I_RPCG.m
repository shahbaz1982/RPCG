% RPCG METHOD FOR SADDLE POINT PROBLEMS WITH 3 X 3
% BLOCK STRUCTURERPCG 
% by SHAHBAZ AHMAD

% Example 1

clear all
close all
clc

p = 16; % Adjust the size as needed

% Matrix W
W = sparse(p*(p+1), p*(p+1));
for i = 1:p*(p+1)
    for j = 1:p*(p+1)
        W(i, j) = exp(-2 * ((i/3)^2 + (j/3)^2));
    end
end

% Identity matrix
I = speye(p*(p+1));

% Matrix A
D2 = diag([ones(1, p^2), 10^(-5) * (1:p^2).^2]);
D3 = diag(10^(-5) * (1:2*p^2).^2);

A = blkdiag(2*W'*W + I, D2, D3);

% Matrix F
F = 2*speye(p+1) - diag(ones(p,1), 1);
F = F(1:end-1, :);

% Matrix E
E = [kron(F, speye(p)); kron(speye(p), F)];

% Matrix B
B = [E, -speye(2*p^2), speye(2*p^2)];

% Matrix C
C = E';

[n na]=size(A);
[m mb]=size(B);
[l lc]=size(C);


Au = [A, B', sparse(n, l); 
      -B, sparse(m, m), -C'; 
      sparse(l, na), C, sparse(l, l)];
Size_of_Coefficient_Matrix_A = size(Au)


% Construct the right-hand side vector b
f = rand(n, 1); 
g = rand(m, 1); 
h = rand(l, 1); 
b = [f; -g; h];


tol = 1e-4; % tolerance for FGMRES

% Initialize FGMRES variables
[Ar Ac]=size(Au);
u0 = sparse(Ar, 1);
max_iter = 1000; % maximum number of iterations for FGMRES

S=B*inv(A)*B';
%---------------------1-PBD Preconditioner---------------------------------------
PBD = [A, sparse(n, m), sparse(n, l); 
      sparse(m, na), S, sparse(m, l); 
      sparse(l, na), sparse(l, m), C*inv(S)*C'];

tic  
[u1, flag1, relres1, iter1, resvec1] = gmres(Au, b, 10, tol, max_iter,PBD);

semilogy(resvec1,'-*');
disp('-----------------------------------------------------')
fprintf('Preconditioner     Flag       RES                 ITER\n')
disp('-----------------------------------------------------')
fprintf('%d                   %d        %d           %d       %d\n',1,flag1,relres1,iter1)
toc
%----------------------2-PSS Preconditioner---------------------------------------
alpha=0.01;
PSS = (1/2)*[alpha*speye(n, na)+ A, B', sparse(n, l); 
      -B, alpha*speye(m, m), -C'; 
      sparse(l, na), C, alpha*speye(l, l)];

tic  
[u2, flag2, relres2, iter2, resvec2] = gmres(Au, b, 10, tol, max_iter,PSS);

hold on
semilogy(resvec2,'-*')
hold off
fprintf('%d                   %d        %d           %d       %d\n',2,flag2,relres2,iter2);
toc
%-----------------------3-PAPSS Preconditioner---------------------------------------
alpha=1;
PAPSS = (1/(2*alpha))*[alpha*speye(n, na)+ A, B', sparse(n, l); 
      -B, alpha*speye(m, m), sparse(m, l); 
      sparse(l, na), sparse(l, m), alpha*speye(l, l)]*[alpha*speye(n, na), sparse(n, m), sparse(n, l); 
      sparse(m, na), alpha*speye(m, m), -C'; 
      sparse(l, na), C, alpha*speye(l, l)];

tic  
[u3, flag3, relres3, iter3, resvec3] = gmres(Au, b, 10, tol, max_iter,PAPSS);
 hold on
 semilogy(resvec3,'-o')
 hold off
fprintf('%d                   %d        %d           %d       %d\n',3,flag3,relres3,iter3)
toc
%--------------------------4-PSRAPSS Preconditioner---------------------------------------
alpha=2.8503;
PSRAPSS = (1/(alpha))*[A, B', sparse(n, l); 
      -B, alpha*speye(m, m), sparse(m, l); 
      sparse(l, na), sparse(l, m), alpha*speye(l, l)]*[alpha*speye(n, na), sparse(n, m), sparse(n, l); 
      sparse(m, na), alpha*speye(m, m), -C'; 
      sparse(l, na), C, sparse(l, l)];

tic  
[u4, flag4, relres4, iter4, resvec4] = gmres(Au, b, 10, tol, max_iter,PSRAPSS);
 hold on
 semilogy(resvec4,'-o')
 hold off
fprintf('%d                   %d        %d           %d       %d\n',4,flag4,relres4,iter4)
toc
%---------------------5-PR Preconditioner---------------------------------------
alpha=0.000001;
PR = [A, B', sparse(n, l); 
      -B, sparse(m, m), -C'; 
      sparse(l, na), C, -alpha*speye(l, l)];

PRN  = [speye(n,na), inv(A)*B', sparse(n, l); 
      sparse(m, na), speye(m, m), -inv(S)*C'; 
      sparse(l, na), sparse(l, m), speye(l, l)];
  
PRM  = [speye(n,na), sparse(n, m), sparse(n, l); 
      -B*inv(A), speye(m, m), sparse(m, l); 
      sparse(l, na), C*inv(S), speye(l, l)];
  
KPR=inv(PRN)*PRM';
tic  
[u5, flag5, relres5, iter5, resvec5] = RPCG(Au, PR, KPR, b, u0, tol, max_iter);
fprintf('%d                   %d        %d           %d\n',5,flag5,relres5,iter5)
toc

hold on
semilogy(resvec5,'-+')

legend('PBD','PSS','PAPSS','PSRAPSS','PR')
%title('Relative Residual Norms')
hold off

disp('-----------------------------------------------------')
disp('1-PBD,2-PSS,3-PAPSS,4-PSRAPSS,5-PR,')
disp('-----------------------------------------------------')


disp('Generating Residuals Graph')
disp ('Generating Eigenvalues Graphs ...')

%--------------------------------Eigenvalues Display------------------------------------------
figure;%A
eigenvalues_A = eig(full(Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PBD^{-1}A
eigenvalues_A = eig(full(inv(PBD)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{BD}^{-1} A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PSS^{-1}A
eigenvalues_A = eig(full(inv(PSS)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{SS}^{-1} A, \alpha = 0.01 ');
xlabel('Real Part');
ylabel('Imaginary Part');


figure;%PAPSS^{-1}A
eigenvalues_A = eig(full(inv(PAPSS)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{APSS}^{-1} A, \alpha = 1 ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PSRAPSS^{-1}A
eigenvalues_A = eig(full(inv(PSRAPSS)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{SRAPSS}^{-1} A, \alpha = 2.8503 ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%PR^{-1}A
eigenvalues_A = eig(full(inv(PR)*Au));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_{R}^{-1} A, \alpha = 0.000001 ');
xlabel('Real Part');
ylabel('Imaginary Part');

%---------------------------------RPCG---------------------------------------------
function [u, flag1, relres1, iter1, resvec1] = RPCG(A, P, K, b, u0, tol, max_iter)
    % Input:
    % A: System matrix
    % P: Preconditioner for Pz = r
    % K: Preconditioner for Kv = z
    % b: Right-hand side vector
    % u0: Initial guess
    % tol: Tolerance for convergence
    % max_iter: Maximum number of iterations
    
    n = size(A, 1);
    m = size(P, 1);
    p = size(K, 1);
    
    u = u0;
    r = b - A * u;
    z = P \ r;
    p = z;
    v = K \ z;
    q = v;
    
    resvec1 = sparse(max_iter, 1);
    
    for iter = 1:max_iter
        alpha = -(v' * r) / (q' * A * p);
        u = u + alpha * p;
        r = r + alpha * A * p;
        
        z = P \ r;
        beta = -(v' * r) / (v' * K * q);
        p = z + beta * p;
        
        v = K \ z;
        q = v + beta * q;
        
        resvec1(iter) = norm(r);
        
        if norm(r) < tol
            break;
        end
    end
    
    % Check convergence
    if norm(r) <= tol
        flag1 = 0; % Convergence achieved
    else
        flag1 = 1; % Maximum number of iterations reached without convergence
    end
    
    relres1 = norm(r) / norm(b); % Relative residual
    iter1 = iter; % Number of iterations
end



