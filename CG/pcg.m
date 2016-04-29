cd /Users/Juraj/Dropbox/USI/PhD/various/GridTools

%size of the problem
N = 40;
ni = N/2; % partitioning in I
nj = N/2; % partitioning in J
nk = N;   % no partitioning in K
npart = 4;

%iteration count
max_iter = 20;

%build global 3D laplace matrix
[~,~,A] = laplacian([N N N]);
%figure(); spy(A); title('Laplacian A');

%set up preconditioner
M = preconditioner(ni,nj,nk);
%figure(); spy(M); title('Preconditioner M');

%set preconditioner to identity to test if the code is the same as naive CG
%M = eye(N^3/4); 

%initial guess
x = zeros((N)^3,1);

%rhs vector
b = ones((N)^3,1);

%residual
r = b - A*x;

%create partitioning
index = @(i,j,k,N) (k-1)*N*N + (j-1)*N + i;
domains = zeros(npart, N^3/npart);
p1 = 1; p2 = 1; p3 = 1; p4 = 1; 
for k = 1:N
    for j = 1:N
        for i = 1:N
            if (i <= N/2 && j <= N/2)    %domain 1
                domains(1,p1) = index(i,j,k,N);
                p1 = p1+1;
            elseif (i <= N/2 && j > N/2) %domain 2
                domains(2,p2) = index(i,j,k,N);
                p2 = p2+1;
            elseif (i > N/2 && j <= N/2) %domain 3
                domains(3,p3) = index(i,j,k,N);
                p3 = p3+1;
            elseif (i > N/2 && j > N/2)  %domain 4
                domains(4,p4) = index(i,j,k,N);
                p4 = p4+1;
            end
        end
    end
end

%precondition each domain
Mr = zeros((N)^3,1);
for i = 1 : npart
    %apply preconditioner
    Mr(domains(i,:)) = M \ r(domains(i,:));
end
rMr = r'*Mr; %reshape(A*d, [3 9])

%search direction 
d = Mr;

%set initial residual
residual = zeros(1,max_iter+1);
residual(1) = norm(b - A*x) / norm(b);

%conjugate gradient iterations
for iter = 1:max_iter
   %step size 
   dAd = d' * A * d;
   alpha = rMr / dAd;
   
   %update solution
   x = x + alpha * d; %reshape(x, [3 9])
   
   %update residual
   r = r - alpha * A * d; %reshape(r, [3 9])
   
   %precondition each domain
   Mr = zeros((N)^3,1);
   for i = 1 : npart
       %apply preconditioner
       Mr(domains(i,:)) = M \ r(domains(i,:));
   end
   rMr_new = r' * Mr;
   
   %orthogonalization parameter
   beta = rMr_new / rMr;
   rMr = rMr_new;
   
   %update search direction
   d = Mr + beta * d; %reshape(d, [3 9])
   
   residual(iter+1) = norm(b - A*x) / norm(b);
end

figure;
plot(1:max_iter+1, residual);
title('Preconditioned Conjugate Gradient convergence');