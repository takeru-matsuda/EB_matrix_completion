% EB with diagonal covariance (Algorithm 2)
function [Mest,SIGMAest,sigma2est,cnt] = EB_diag(X,obs,sigma2,EPS_ll,EPS_M)
%
% Input:
%    X:     data matrix (p times q)
%           please set unobserved entries to arbitrary value (e.g., zero)
%    obs:   observation flag (p times q)
%           1 -> observed
%           0 -> unobserved
%   sigma2: initial values for sigma_1^2, ..., sigma_q^2 (1 times q)
%   EPS_ll: tolerance for log-likelihood (epsilon_1)
%   EPS_M:  tolerance for update of M (epsilon_2)
%
% Output:
%   Mest:       estimate of M (p times q)
%   SIGMAest:   estimate of SIGMA (q times q)
%   sigma2est:  estimate of sigma_1^2, ..., sigma_q^2 (1 times q)
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
        sumR = zeros(1,q);
        Rs = zeros(q,q);
        for i=1:p
            nobs = sum(obs(i,:));
            b = (X(i,:).*obs(i,:))';
            obs_loc = find(obs(i,:));
            tmp = inv(diag(sigma2est(obs_loc))+SIGMAest(obs_loc,obs_loc));
            tmp2 = zeros(q,q);
            tmp2(obs_loc,obs_loc) = tmp;
            R = SIGMAest-SIGMAest*tmp2*SIGMAest;
            Mest(i,:) = (R*(b./sigma2est'))';
            Rs = Rs+R;
            sumR = sumR+obs(i,:).*diag(R)';
        end
        SIGMAest = (Mest'*Mest+Rs)/p;
        for j=1:q
            nobs = sum(obs(:,j));
            sigma2est(j) = (norm(X(obs(:,j)==1,j)-Mest(obs(:,j)==1,j))^2+sumR(j))/nobs;
        end
        ll = 0;
        for i=1:p
            tmp = obs(i,:);
            SIGMAsub = SIGMAest(tmp==1,tmp==1);
            ll = ll-log(det(SIGMAsub+diag(sigma2est(tmp==1))))/2-X(i,tmp==1)*((SIGMAsub+diag(sigma2est(tmp==1)))\X(i,tmp==1)')/2;
        end
        if cnt > 1 && ll-prev_ll < EPS_ll
            break;
        end
        prev_ll = ll;
    end
end

