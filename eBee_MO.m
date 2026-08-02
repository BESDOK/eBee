function out = eBee_MO(ObjFnc, userdata, N, D, lb, ub, MaxCycle, ArchiveSize)
% eBee_MO  v2 : Multi-objective eBee (Besdok) with an improved Deb's rule.
%
% The eBee search mechanism is preserved EXACTLY as in eBee_PureCode.m:
%   Direction-Search (T=0.10), Area-Search (T=0.90), Random-Search (p=0.10),
%   CSize=2 Meadow bookkeeping, wildflowers refresh every 100 cycles, and the
%   untouched generateMS / generateMask / levy_flight / SelectByProbability.
%
% Only the SELECTION layer is multi-objective. Changes over v1:
%   (1) Every offspring produced by an eBee phase is offered to the external
%       archive, even when Deb's rule keeps the parent in the population.
%       In v1 a non-dominated offspring rejected by the coin-flip tie-break
%       was lost forever, which cost front density (|PF|) and hurt IGD/HV.
%   (2) The mutually-non-dominated tie is no longer a blind 0.5 coin flip.
%       It is resolved by a density criterion in objective space: whichever
%       of the two sits in the sparser region of the current archive wins.
%       This turns the tie-break into a diversity pressure instead of noise,
%       and it is deterministic apart from an exact-equality fallback.
%   (3) Deb's rule now uses epsilon-dominance (eps = 0 by default, set
%       ARCHIVE_EPS below to a small positive value for a coarser, evenly
%       spread front) so that near-identical solutions do not both survive.
%   (4) The archive truncation keeps the extreme points (infinite crowding
%       distance) unconditionally, so the front ends are never eroded.
%
% Deb's rule, pairwise (candidate Y vs incumbent X):
%   1) one feasible, one not     -> the feasible one wins
%   2) both infeasible           -> smaller total constraint violation wins
%   3) both feasible             -> Pareto (epsilon-)dominance decides
%   4) both feasible, non-dominated -> sparser archive region wins
%
% INPUT
%   ObjFnc      : f = ObjFnc(P,userdata) -> (N x M); optionally [f,cv] with
%                 cv (N x 1) total constraint violation, cv <= 0 = feasible
%   userdata    : passed straight to ObjFnc
%   N, D        : colony size, problem dimension
%   lb, ub      : scalar or 1 x D bounds
%   MaxCycle    : number of cycles
%   ArchiveSize : optional, default 100
%
% OUTPUT
%   out.archive / out.archiveFit / out.archiveCV : archive and Pareto front
%   out.gbest / out.gval                         : last leader

if nargin < 8 || isempty(ArchiveSize), ArchiveSize = 100; end

ARCHIVE_EPS = 0;    % epsilon for epsilon-dominance (0 = classical dominance)

CSize = 2;

Meadow = rand(CSize*N, D) .* (ub-lb) + lb;
[fitMeadow, cvMeadow] = evalPop(ObjFnc, Meadow, userdata);

[Archive, ArchiveFit, ArchiveCV] = updateArchive([], [], [], Meadow, fitMeadow, cvMeadow, ArchiveSize, ARCHIVE_EPS);
[gBest, gBestVal] = selectLeader(Archive, ArchiveFit);

wildflowers = lb + (ub -lb) .* rand(N, D);

J = randperm(CSize*N, N);

for iteration = 1:MaxCycle

    % Initialization
    FlowerField = Meadow(J,:);
    fitFlowers  = fitMeadow(J,:);
    cvFlowers   = cvMeadow(J);

    Flowers0    = FlowerField;
    fitFlowers0 = fitFlowers;
    cvFlowers0  = cvFlowers;

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
    [FlowerField, fitFlowers, cvFlowers, Yb, fYb, cvYb] = ...
        Update(FlowerField, fitFlowers, cvFlowers, EmployedBee, lb, ub, ObjFnc, userdata, ArchiveFit, ARCHIVE_EPS);
    % improvement (1): offered to the archive regardless of the population outcome
    [Archive, ArchiveFit, ArchiveCV] = updateArchive(Archive, ArchiveFit, ArchiveCV, Yb, fYb, cvYb, ArchiveSize, ARCHIVE_EPS);


    % Area-Search
    while 1, j0 = randperm(N); if ( sum(j0 == 1 : N ) == 0 ), break; end, end
    while 1, j1 = randperm(N); if ( sum(j1 == j0 ) == 0 ), break; end, end

    T=0.90;
    [M,scale] = generateMS(N, D, T);

    % eBee's probabilistic selection operator is unchanged; it is fed with a
    % scalar quality consistent with Deb's rule (feasibility -> Pareto rank
    % -> crowding distance).
    ind=SelectByProbability(scalarQuality(fitFlowers0,cvFlowers0),ceil(N/2));
    indFlowers=ind(randi(numel(ind),1,N));

    if rand < rand, j1=indFlowers; end

    dx = Flowers0(j1,:) - FlowerField(j0,:);
    OnlookerFlowers = FlowerField + M .* scale .* dx;

    [nectar1, fitnectar1, cvnectar1, Yo, fYo, cvYo] = ...
        Update(Flowers0, fitFlowers0, cvFlowers0, OnlookerFlowers, lb, ub, ObjFnc, userdata, ArchiveFit, ARCHIVE_EPS);
    [Archive, ArchiveFit, ArchiveCV] = updateArchive(Archive, ArchiveFit, ArchiveCV, Yo, fYo, cvYo, ArchiveSize, ARCHIVE_EPS);

    j = debBetter(fitnectar1, cvnectar1, fitFlowers, cvFlowers, ArchiveFit, ARCHIVE_EPS);
    FlowerField(j,:) = nectar1(j,:);
    fitFlowers(j,:)  = fitnectar1(j,:);
    cvFlowers(j)     = cvnectar1(j);

    % Random-Search
    if rand < 0.10
        [M,scale] = generateMS(N, D,0.90);
        dx = wildflowers - FlowerField;
        ScoutBee = FlowerField + M .* scale .* dx;
        [FlowerField, fitFlowers, cvFlowers, Ys, fYs, cvYs] = ...
            Update(FlowerField, fitFlowers, cvFlowers, ScoutBee, lb, ub, ObjFnc, userdata, ArchiveFit, ARCHIVE_EPS);
        [Archive, ArchiveFit, ArchiveCV] = updateArchive(Archive, ArchiveFit, ArchiveCV, Ys, fYs, cvYs, ArchiveSize, ARCHIVE_EPS);
    end

    % Global Update #1
    Meadow(J,:)    = FlowerField;
    fitMeadow(J,:) = fitFlowers;
    cvMeadow(J)    = cvFlowers;
    if rand<0.50, J=randperm(CSize*N,N); end

    % Global Update for global solutions (archive + leader)
    [Archive, ArchiveFit, ArchiveCV] = updateArchive(Archive, ArchiveFit, ArchiveCV, FlowerField, fitFlowers, cvFlowers, ArchiveSize, ARCHIVE_EPS);
    [gBest, gBestVal] = selectLeader(Archive, ArchiveFit);

    out.archive    = Archive;
    out.archiveFit = ArchiveFit;
    out.archiveCV  = ArchiveCV;
    out.gbest      = gBest;
    out.gval       = gBestVal;
    assignin('base','eBee_MO_out',out);

    fprintf('Iter=%d -->  |PF|=%d \n', iteration, size(ArchiveFit,1));

    if mod(iteration,100)==0, wildflowers = lb + (ub -lb) .* rand(N, D); end
end

end


% ======================= Deb's rule greedy selection ======================
function [X, fitX, cvX, Y, fitY, cvY] = Update(X, fitX, cvX, Y, lb, ub, fObj, userdata, AF, eps0)
Y = max(lb, min(ub, Y));
[fitY, cvY] = evalPop(fObj, Y, userdata);
ind = debBetter(fitY, cvY, fitX, cvX, AF, eps0);
X(ind,:)    = Y(ind,:);
fitX(ind,:) = fitY(ind,:);
cvX(ind)    = cvY(ind);
end


function ind = debBetter(fY, cvY, fX, cvX, AF, eps0)
% Row-wise Deb's rule: true where candidate Y replaces incumbent X.
if nargin < 5, AF = []; end
if nargin < 6, eps0 = 0; end

n   = size(fY,1);
ind = false(n,1);

feasY = cvY <= 0;
feasX = cvX <= 0;

% Rule 1: exactly one of the two is feasible
ind(feasY & ~feasX) = true;

% Rule 2: both infeasible -> smaller total violation wins
k = ~feasY & ~feasX;
ind(k) = cvY(k) < cvX(k);

% Rules 3-4: both feasible
k = find(feasY & feasX);
if isempty(k), return; end

% Rule-4 material: archive dominance status first, sparsity second.
domY = dominatedByArchive(fY(k,:), AF, eps0);
domX = dominatedByArchive(fX(k,:), AF, eps0);
[dY, dX] = archiveSparsity(fY(k,:), fX(k,:), AF);

for t = 1:numel(k)
    i = k(t);
    a = fY(i,:); b = fX(i,:);
    if all(a - eps0 <= b) && any(a + eps0 < b)
        ind(i) = true;                    % Y (epsilon-)dominates X
    elseif all(b - eps0 <= a) && any(b + eps0 < a)
        ind(i) = false;                   % X (epsilon-)dominates Y
    elseif domY(t) && ~domX(t)
        ind(i) = false;                   % Y lags the archive -> keep X
    elseif domX(t) && ~domY(t)
        ind(i) = true;                    % X lags the archive -> take Y
    else
        % Both equally placed w.r.t. the archive: sparser region wins.
        if dY(t) > dX(t)
            ind(i) = true;
        elseif dY(t) < dX(t)
            ind(i) = false;
        else
            ind(i) = rand < 0.50;         % exact tie only
        end
    end
end
end


function dom = dominatedByArchive(F, AF, eps0)
% True where a point is (epsilon-)dominated by at least one archive member.
n = size(F,1);
dom = false(n,1);
if isempty(AF), return; end
for i = 1:n
    a = F(i,:);
    dom(i) = any( all(bsxfun(@le, AF - eps0, a), 2) & any(bsxfun(@lt, AF + eps0, a), 2) );
end
end


function [dY, dX] = archiveSparsity(FY, FX, AF)
% Normalised distance from each point to its nearest archive member.
% Larger value = sparser (more valuable) region of objective space.
nY = size(FY,1);
if isempty(AF) || size(AF,1) < 2
    dY = zeros(nY,1); dX = zeros(nY,1); return;
end
lo  = min(AF,[],1);
hi  = max(AF,[],1);
rng = hi - lo; rng(rng == 0) = 1;

An = bsxfun(@rdivide, bsxfun(@minus, AF, lo), rng);
Yn = bsxfun(@rdivide, bsxfun(@minus, FY, lo), rng);
Xn = bsxfun(@rdivide, bsxfun(@minus, FX, lo), rng);

dY = nearestDist(Yn, An);
dX = nearestDist(Xn, An);
end


function d = nearestDist(P, A)
n = size(P,1);
d = zeros(n,1);
for i = 1:n
    df = bsxfun(@minus, A, P(i,:));
    d(i) = min(sqrt(sum(df.^2, 2)));
end
end


% ============================ archive handling ============================
function [A, AF, ACV] = updateArchive(A, AF, ACV, X, F, CV, maxSize, eps0)
if nargin < 8, eps0 = 0; end

feas = CV <= 0;
if any(feas)
    X = X(feas,:); F = F(feas,:); CV = CV(feas);
elseif isempty(AF)
    % No feasible solution found yet: keep the least-violating individuals.
    [~, o] = sort(CV); o = o(1:min(maxSize,numel(o)));
    X = X(o,:); F = F(o,:); CV = CV(o);
else
    return;
end

A   = [A;   X];
AF  = [AF;  F];
ACV = [ACV; CV];

[AF, iu] = unique(AF, 'rows', 'stable');
A = A(iu,:); ACV = ACV(iu);

nd = nonDominatedMask(AF, eps0);
A = A(nd,:); AF = AF(nd,:); ACV = ACV(nd);

if size(AF,1) > maxSize
    % Improvement (4): the extreme points are retained unconditionally.
    cd   = crowdingDistance(AF);
    keep = find(isinf(cd));
    rest = find(~isinf(cd));
    [~, o] = sort(cd(rest), 'descend');
    room = maxSize - numel(keep);
    if room > 0
        keep = [keep; rest(o(1:min(room, numel(rest))))];
    else
        keep = keep(1:maxSize);
    end
    keep = sort(keep);
    A = A(keep,:); AF = AF(keep,:); ACV = ACV(keep);
end
end


function [g, gval] = selectLeader(A, AF)
% Binary tournament on crowding distance: the leader comes from the least
% crowded region, which pushes the front outwards instead of inwards.
K = size(A,1);
if K == 1, g = A(1,:); gval = AF(1,:); return; end
cd = crowdingDistance(AF);
i1 = randi(K); i2 = randi(K);
if cd(i1) >= cd(i2), i = i1; else, i = i2; end
g = A(i,:); gval = AF(i,:);
end


function nd = nonDominatedMask(F, eps0)
if nargin < 2, eps0 = 0; end
K  = size(F,1);
nd = true(K,1);
for i = 1:K
    if ~nd(i), continue; end
    a = F(i,:);
    dom = all(bsxfun(@le, a - eps0, F), 2) & any(bsxfun(@lt, a + eps0, F), 2);
    nd(dom) = false;
end
end


function cd = crowdingDistance(F)
[K, Mo] = size(F);
cd = zeros(K,1);
if K <= 2, cd = inf(K,1); return; end
for m = 1:Mo
    [s, o] = sort(F(:,m));
    cd(o(1)) = inf; cd(o(end)) = inf;
    rg = s(end) - s(1);
    if rg == 0, continue; end
    cd(o(2:end-1)) = cd(o(2:end-1)) + (s(3:end) - s(1:end-2)) / rg;
end
end


function q = scalarQuality(F, CV)
% Scalar quality consistent with Deb's rule, used only to feed the unchanged
% SelectByProbability operator. Lower value = better.
n = size(F,1);
q = zeros(n,1);
feas = CV <= 0;

if any(feas)
    Ff  = F(feas,:);
    r   = paretoRank(Ff);
    cd  = crowdingDistance(Ff);
    cdn = cd;
    fin = ~isinf(cdn);
    if any(fin), cdn(~fin) = max(cdn(fin)); else, cdn(:) = 1; end
    if max(cdn) > 0, cdn = cdn / max(cdn); end
    q(feas) = r - 0.5*cdn;
end
if any(~feas)
    base = max([q(feas); 0]) + 1;
    v = CV(~feas);
    if max(v) > min(v), v = (v - min(v)) / (max(v) - min(v)); else, v = zeros(size(v)); end
    q(~feas) = base + v;
end
end


function r = paretoRank(F)
K = size(F,1);
r = zeros(K,1);
remaining = true(K,1);
front = 1;
while any(remaining)
    idx = find(remaining);
    nd  = nonDominatedMask(F(idx,:));
    r(idx(nd)) = front;
    remaining(idx(nd)) = false;
    front = front + 1;
end
end


function [F, CV] = evalPop(fObj, X, userdata)
% Supports both f = ObjFnc(X,userdata) and [f,cv] = ObjFnc(X,userdata).
try
    [F, CV] = feval(fObj, X, userdata);
catch
    F  = feval(fObj, X, userdata);
    CV = zeros(size(X,1),1);
end
if isempty(CV), CV = zeros(size(X,1),1); end
CV = CV(:);
end


% ============ eBee internal operators - UNCHANGED from eBee_PureCode ======
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
