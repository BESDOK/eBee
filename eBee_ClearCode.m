function out = eBee_PureCode(ObjFnc, userdata, N, D, lb, ub, MaxCycle)
CSize=2;

Meadow = rand(CSize*N, D) .* (ub-lb) +lb;
fitMeadow = feval(ObjFnc,Meadow,userdata);
[fitTheBestFlowers, idx] = min(fitMeadow);
gBest = Meadow(idx, :);
gBestVal = fitTheBestFlowers;
wildflowers = lb + (ub -lb) .* rand(N, D);

J=randperm(CSize*N,N);

for iteration = 1:MaxCycle

    % Initialization
    FlowerField=Meadow(J,:);
    fitFlowers=fitMeadow(J);

    Flowers0=FlowerField;
    fitFlowers0=fitFlowers;

    % Direction-Search
    while 1, j=randperm(N); if sum(j==1:N)==0, break; end, end
    T=0.10;
    [M,scale] = generateMS(N, D, T);

    if rand < 0.50
        dx = FlowerField(j,:) - FlowerField;
    else
        idx = (1:N)';

        j1 = randi(N-1, N, 1);
        j1(j1 >= idx) = j1(j1 >= idx) + 1;

        j2 = randi(N-2, N, 1);
        min_idx = min(idx, j1);
        max_idx = max(idx, j1);

        j2(j2 >= min_idx) = j2(j2 >= min_idx) + 1;
        j2(j2 >= max_idx) = j2(j2 >= max_idx) + 1;

        w1 = rand(N,1) ;
        w2 = rand(N,1) ;
        k = w1 + w2 ;
        j = k > 1 ;
        w1(j==1) = 1 - w1(j==1); w2(j==1) = 1 - w2(j==1);
        if rand < 0.50, hField=FlowerField; else, hField=gBest ; end
        dx = w1 .* (FlowerField(j1,:)-hField) + w2 .* (FlowerField(j2,:)- hField) ;
    end

    EmployedBee = FlowerField + M .* scale .* dx;
    [FlowerField, fitFlowers] = Update(FlowerField, fitFlowers, EmployedBee, lb, ub, ObjFnc, userdata);


    % Area-Search    
    while 1, j0 = randperm(N); if ( sum(j0 == 1 : N ) == 0 ), break; end, end
    while 1, j1 = randperm(N); if ( sum(j1 == j0 ) == 0 ), break; end, end

    T=0.90;
    [M,scale] = generateMS(N, D, T);

    ind=SelectByProbability(fitFlowers0,ceil(N/2));
    indFlowers=ind(randi(numel(ind),1,N));

    if rand < rand, j1=indFlowers; end

    dx = Flowers0(j1,:) - FlowerField(j0,:);
    OnlookerFlowers = FlowerField + M .* scale .* dx;

    [nectar1, fitnectar1] = Update(Flowers0, fitFlowers0, OnlookerFlowers, lb, ub, ObjFnc, userdata);

    j=fitnectar1<fitFlowers;
    FlowerField(j,:)=nectar1(j,:);
    fitFlowers(j)=fitnectar1(j);

    % Random-Search
    if rand < 0.10
        [M,scale] = generateMS(N, D,0.90);
        dx = wildflowers - FlowerField;
        ScoutBee = FlowerField + M .* scale .* dx;
        [FlowerField, fitFlowers] = Update(FlowerField, fitFlowers, ScoutBee, lb, ub, ObjFnc, userdata);
    end

    % Global Update #1
    Meadow(J,:)=FlowerField;
    fitMeadow(J)=fitFlowers;
    if rand<0.50, J=randperm(CSize*N,N); end

    % Global Update for Global solutions (gBest, and gBestVal)
    [fBest, idx] = min(fitMeadow);
    if fBest < gBestVal, gBestVal = fBest; gBest = Meadow(idx,:); end

    out.gbest = gBest;
    out.gval = gBestVal;
    assignin('base','eBee_out',out);

fprintf('Iter=%d -->  fMin=%5.16f \n', iteration, gBestVal);

if mod(iteration,100)==0, wildflowers = lb + (ub -lb) .* rand(N, D); end
end

end


function [X, fitX] = Update(X, fitX, Y, lb, ub, fObj, userdata)
Y = max(lb, min(ub, Y));
fitY = feval(fObj, Y, userdata);
ind = fitY < fitX;
X(ind,:) = Y(ind,:);
fitX(ind) = fitY(ind);
end

function [M, scale] = generateMS(nnectar, nDim, T)
M = generateMask(nnectar, nDim);
if rand <= T
    c = 1;
else
    c = nDim;
end
beta = 1.10 + rand*0.90;
scale = levy_flight(nnectar, c, beta);
end


function M = generateMask(A, B)
M=zeros(A,B);
k=abs(randi([0 1],1) - rand.^randi([2 10]) );
for i=1:A
    v=randperm(B);
    h=ceil(k*B);
    M(i,v(1:h))=1;
end
end


function ind=SelectByProbability(x,N)
y=abs(x+min(x)+eps);
f=y.^5;
p=f./sum(f);
ind=nan(N,1);
for i=1:N
    [~,ind(i)]=min( abs( p - rand^randi([5 10]) ) );
end
end

function scale = levy_flight(N, ndim, beta)
num = gamma(1 + beta) * sin(pi * beta / 2);
den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
sigma_u = (num / den)^(1 / beta);

U = sigma_u * randn(N, ndim);
V = randn(N, ndim);

scale = U ./ (abs(V) .^ (1 / beta));

k = [ones(1,ceil(0.10*N)) 3*rand(1,ceil(0.90*N)) ];
k = k(randperm(N))';
h = sign(scale);
scale = h .* abs(scale) .^ k;

end