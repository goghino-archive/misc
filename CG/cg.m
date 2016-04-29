cd /Users/Juraj/Dropbox/USI/PhD/various/GridTools

%size of the problem
N = 40;

%iteration count
max_iter = 20;

%build 3D laplace matrix
[~,~,A] = laplacian([N N N]);
%spy(A)

%initial guess
x = zeros((N)^3,1);

%rhs vector
b = ones((N)^3,1);

%residual
r = b - A*x;

%search direction
d = r;

%set initial residual
residual = zeros(1,max_iter+1);
residual(1) = norm(b - A*x) / norm(b);

%conjugate gradient iterations
for iter = 1:max_iter
   rr = r'*r; %reshape(A*d, [3 9])
   dAd = d' * A * d;
   
   %step size
   alpha = rr / dAd;
   
   %update solution
   x = x + alpha * d; %reshape(x, [3 9])
   
   %update residual
   r = r - alpha * A * d; %reshape(r, [3 9])
   
   %orthogonalization parameter
   beta = r' * r / rr;
   
   %new search direction
   d = r + beta * d; %reshape(d, [3 9])
   
   residual(iter+1) = norm(b - A*x) / norm(b);
end

figure;
plot(1:max_iter+1, residual);
title('Conjugate Gradient convergence');
