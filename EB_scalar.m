% EB with scalar covariance (Algorithm 1)
function [Mest,SIGMAest,sigma2est,cnt] = EB_scalar(X,obs,sigma2,EPS_ll,EPS_M)
%
% Input:
%    X:     data matrix (p times q)
%           please set unobserved entries to arbitrary value (e.g., zero)
%    obs:   observation flag (p times q)
%           1 -> observed
%           0 -> unobserved
%   sigma2: initial value for sigma^2 (scalar)
%   EPS_ll: tolerance for log-likelihood (epsilon_1)
%   EPS_M:  tolerance for update of M (epsilon_2)
%
% Output:
%   Mest:       estimate of M (p times q)
%   SIGMAest:   estimate of SIGMA (q times q)
%   sigma2est:  estimate of sigma^2 (scalar)
%   cnt:        number of iterations until convergence (scalar)
%
    p = size(X,1);
    q = size(X,2);
    Mest = zeros(p,q);
    for i=1:p
        for j=1:q
            if obs(i,j)==1
                Mest(i,j) = X(i,j);
            end
        end
    end
    Mprev = Mest;
    SIGMAest = (Mprev'*Mprev)/p;
    sigma2est = sigma2;
    cnt = 0;
    prev_ll = -inf;
    while cnt == 0 || norm(Mest-Mprev,'fro')/norm(Mprev,'fro') > EPS_M
        cnt = cnt+1;
        Mprev = Mest;
        sumR = 0;
        Rs = zeros(q,q);
        for i=1:p
            nobs = sum(obs(i,:));
            b = (X(i,:).*obs(i,:))';
            obs_loc = find(obs(i,:));
            tmp = inv(sigma2est*eye(nobs)+SIGMAest(obs_loc,obs_loc));
            tmp2 = zeros(q,q);
            tmp2(obs_loc,obs_loc) = tmp;
            R = SIGMAest-SIGMAest*tmp2*SIGMAest;
            Mest(i,:) = (R*b)'/sigma2est;
            Rs = Rs+R;
            sumR = sumR+trace(R(obs_loc,obs_loc));
        end
        SIGMAest = (Mest'*Mest+Rs)/p;
        sigma2est = (norm(X(obs==1)-Mest(obs==1))^2+sumR)/nnz(obs);
        ll = 0;
        for i=1:p
            tmp = obs(i,:);
            SIGMAsub = SIGMAest(tmp==1,tmp==1);
            ll = ll-log(det(SIGMAsub+sigma2est*eye(nnz(tmp))))/2-X(i,tmp==1)*((SIGMAsub+sigma2est*eye(nnz(tmp)))\X(i,tmp==1)')/2;
        end
        if cnt > 1 && ll-prev_ll < EPS_ll
            break;
        end
        prev_ll = ll;
    end
end

