function [M] = preconditioner(ni,nj,nk)
%constructs jacobi block diagonal preconditioner
%size of the local domain (ni,nj,nk), natural ordering

%preconditioner dimension
N = ni*nj*nk;

M = sparse(N,N);

index = @(i,j,k,ni,nj) (k-1)*nj*ni + (j-1)*ni + i;


for i = 1:ni
    for j = 1:nj
        for k = 1:nk  
            %diagonal element
            M(index(i,j,k,ni,nj),index(i,j,k,ni,nj)) = 6;

            %N,S,E,W neighbours
            if (i+1 <= ni)
                M(index(i,j,k,ni,nj),index(i+1,j,k,ni,nj)) = -1;
            end
            if (i-1 >= 1)
                M(index(i,j,k,ni,nj),index(i-1,j,k,ni,nj)) = -1;
            end
            if (j+1 <= nj)
                M(index(i,j,k,ni,nj),index(i,j+1,k,ni,nj)) = -1;
            end
            if (j-1 >= 1)
                M(index(i,j,k,ni,nj),index(i,j-1,k,ni,nj)) = -1;
            end
            
            %up and down neighbours
            if (k+1 <= nk)
                M(index(i,j,k,ni,nj),index(i,j,k+1,ni,nj)) = -1;
            end
            if (k-1 >= 1)
                M(index(i,j,k,ni,nj),index(i,j,k-1,ni,nj)) = -1;
            end
        end
    end
end

end