function out = eBee_MO_english(ObjFnc, userdata, N, D, lb, ub, MaxCycle, ArchiveSize)
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
%                 The function must be vectorized: one row of P per candidate
%                 solution, one row of f per candidate. M is the number of
%                 objectives and is inferred automatically from the first call.
%   userdata    : passed straight to ObjFnc (struct, matrix, cell, ... or [])
%   N, D        : colony size, problem dimension. N >= 3 is required because
%                 the Direction-Search draws two partners distinct from the
%                 current index via randi(N-2).
%   lb, ub      : scalar or 1 x D bounds; used for initialization and for
%                 saturation clamping of every trial solution.
%   MaxCycle    : number of cycles. Each cycle costs at most three vectorized
%                 objective calls of N rows (Direction, Area, and Random when
%                 it triggers).
%   ArchiveSize : optional, default 100. Upper bound on the external Pareto
%                 archive; truncation is by crowding distance.
%
% OUTPUT
%   out.archive / out.archiveFit / out.archiveCV : archive and Pareto front
%   out.gbest / out.gval                         : last leader
%
% The output struct is also mirrored to the base workspace as 'eBee_MO_out'
% at every cycle, so a long run can be monitored or aborted without losing
% the current front. All randomness comes from rand/randi/randn, so a run is
% reproducible by seeding (e.g. rng(0,'twister')) before the call.

if nargin < 8 || isempty(ArchiveSize), ArchiveSize = 100; end

% Epsilon used by every dominance test in this file (Deb's rule, archive
% filtering, and the archive-lag check). 0 reproduces classical Pareto
% dominance; a small positive value produces a coarser but more evenly spread
% front by preventing near-identical points from both surviving.
ARCHIVE_EPS = 0;

% Colony size multiplier: the Meadow stores CSize*N solutions while only N of
% them (the FlowerField) are evolved in any single cycle.
CSize = 2;

% --- Meadow initialization -----------------------------------------------
Meadow = rand(CSize*N, D) .* (ub-lb) + lb;
[fitMeadow, cvMeadow] = evalPop(ObjFnc, Meadow, userdata);

% The external archive is seeded from the whole initial Meadow, then the first
% leader is drawn from it.
[Archive, ArchiveFit, ArchiveCV] = updateArchive([], [], [], Meadow, fitMeadow, cvMeadow, ArchiveSize, ARCHIVE_EPS);
[gBest, gBestVal] = selectLeader(Archive, ArchiveFit);

% Auxiliary pool of purely random, never-evaluated points used only as a
% direction source by the Scout-Bee phase; refreshed every 100 cycles.
wildflowers = lb + (ub -lb) .* rand(N, D);

% Indices of the Meadow members forming the current active FlowerField.
J = randperm(CSize*N, N);

for iteration = 1:MaxCycle

    % Initialization
    % Copy the active subset out of the Meadow (positions, objectives, and
    % constraint violations travel together throughout the cycle).
    FlowerField = Meadow(J,:);
    fitFlowers  = fitMeadow(J,:);
    cvFlowers   = cvMeadow(J);

    % Frozen cycle-start snapshot: the Area-Search compares against this
    % unchanging reference instead of the already-updated FlowerField.
    Flowers0    = FlowerField;
    fitFlowers0 = fitFlowers;
    cvFlowers0  = cvFlowers;

    % Direction-Search
    % Derangement of 1:N (no fixed point) so no flower is paired with itself.
    while 1, j=randperm(N); if sum(j==1:N)==0, break; end, end

    % T is the probability of a single shared Levy scale per individual.
    % T=0.10 means per-dimension (anisotropic) scales 90% of the time, which
    % makes the employed phase strongly exploratory.
    T=0.10;
    [M,scale] = generateMS(N, D, T);

    if rand < 0.50
        % Variant A: plain pairwise differential vector via the derangement.
        dx = FlowerField(j,:) - FlowerField;
    else
        % Variant B: convex two-partner recombination toward a reference field.
        idx = (1:N)';

        % j1 ~ uniform over 1:N excluding i. Drawing in 1:N-1 and shifting
        % whenever j1 >= i avoids a rejection loop.
        j1 = randi(N-1, N, 1);
        j1(j1 >= idx) = j1(j1 >= idx) + 1;

        % j2 ~ uniform over 1:N excluding both i and j1: draw in 1:N-2 and
        % shift past the smaller, then the larger, of the two excluded values.
        j2 = randi(N-2, N, 1);
        min_idx = min(idx, j1);
        max_idx = max(idx, j1);

        j2(j2 >= min_idx) = j2(j2 >= min_idx) + 1;
        j2(j2 >= max_idx) = j2(j2 >= max_idx) + 1;

        % Triangle folding makes w1 + w2 <= 1, i.e. (w1,w2) uniform on the
        % simplex, which keeps the combined step bounded.
        w1 = rand(N,1) ;
        w2 = rand(N,1) ;
        k = w1 + w2 ;
        j = k > 1 ;
        w1(j==1) = 1 - w1(j==1); w2(j==1) = 1 - w2(j==1);

        % Reference: the population itself (exploratory) or the archive leader
        % gBest (exploitative, all rows pulled toward the current leader).
        if rand < 0.50, hField=FlowerField; else, hField=gBest ; end
        dx = w1 .* (FlowerField(j1,:)-hField) + w2 .* (FlowerField(j2,:)- hField) ;
    end

    % Only the dimensions selected by the binary mask M are perturbed, with a
    % Levy-distributed magnitude.
    EmployedBee = FlowerField + M .* scale .* dx;

    % Update returns BOTH the surviving population and the raw evaluated
    % offspring (Yb, fYb, cvYb), so the offspring can still reach the archive.
    [FlowerField, fitFlowers, cvFlowers, Yb, fYb, cvYb] = ...
        Update(FlowerField, fitFlowers, cvFlowers, EmployedBee, lb, ub, ObjFnc, userdata, ArchiveFit, ARCHIVE_EPS);
    % improvement (1): offered to the archive regardless of the population outcome
    [Archive, ArchiveFit, ArchiveCV] = updateArchive(Archive, ArchiveFit, ArchiveCV, Yb, fYb, cvYb, ArchiveSize, ARCHIVE_EPS);


    % Area-Search
    % j0 is a derangement of 1:N; j1 is a permutation differing from j0 in
    % every position. The pair (j1,j0) is therefore index-wise distinct, which
    % rules out degenerate zero difference vectors.
    while 1, j0 = randperm(N); if ( sum(j0 == 1 : N ) == 0 ), break; end, end
    while 1, j1 = randperm(N); if ( sum(j1 == j0 ) == 0 ), break; end, end

    % T=0.90: a single shared Levy scale per individual 90% of the time, giving
    % coherent whole-vector moves that suit the exploitative onlooker phase.
    T=0.90;
    [M,scale] = generateMS(N, D, T);

    % eBee's probabilistic selection operator is unchanged; it is fed with a
    % scalar quality consistent with Deb's rule (feasibility -> Pareto rank
    % -> crowding distance).
    ind=SelectByProbability(scalarQuality(fitFlowers0,cvFlowers0),ceil(N/2));

    % The ceil(N/2) recruited indices are resampled with replacement to length
    % N (the waggle-dance recruitment of onlooker bees).
    indFlowers=ind(randi(numel(ind),1,N));

    % rand < rand is a fair coin (p = 1/2): half the time the elite-biased
    % indices replace the random permutation j1.
    if rand < rand, j1=indFlowers; end

    dx = Flowers0(j1,:) - FlowerField(j0,:);
    OnlookerFlowers = FlowerField + M .* scale .* dx;

    % The onlooker trials first compete against the frozen snapshot Flowers0,
    % producing an independent candidate set (nectar1) rather than overwriting
    % the live FlowerField.
    [nectar1, fitnectar1, cvnectar1, Yo, fYo, cvYo] = ...
        Update(Flowers0, fitFlowers0, cvFlowers0, OnlookerFlowers, lb, ub, ObjFnc, userdata, ArchiveFit, ARCHIVE_EPS);
    [Archive, ArchiveFit, ArchiveCV] = updateArchive(Archive, ArchiveFit, ArchiveCV, Yo, fYo, cvYo, ArchiveSize, ARCHIVE_EPS);

    % Merge step: a nectar solution enters the live field only where Deb's rule
    % declares it better, i.e. a second comparison front.
    j = debBetter(fitnectar1, cvnectar1, fitFlowers, cvFlowers, ArchiveFit, ARCHIVE_EPS);
    FlowerField(j,:) = nectar1(j,:);
    fitFlowers(j,:)  = fitnectar1(j,:);
    cvFlowers(j)     = cvnectar1(j);

    % Random-Search
    % Triggered with probability 0.10 per cycle; the main stagnation-escape
    % mechanism, moving flowers toward the random wildflower pool.
    if rand < 0.10
        [M,scale] = generateMS(N, D,0.90);
        dx = wildflowers - FlowerField;
        ScoutBee = FlowerField + M .* scale .* dx;
        [FlowerField, fitFlowers, cvFlowers, Ys, fYs, cvYs] = ...
            Update(FlowerField, fitFlowers, cvFlowers, ScoutBee, lb, ub, ObjFnc, userdata, ArchiveFit, ARCHIVE_EPS);
        [Archive, ArchiveFit, ArchiveCV] = updateArchive(Archive, ArchiveFit, ArchiveCV, Ys, fYs, cvYs, ArchiveSize, ARCHIVE_EPS);
    end

    % Global Update #1
    % Write the evolved subset back into the Meadow. With probability 0.50 a
    % fresh active subset is drawn; keeping J otherwise lets a promising subset
    % be refined over consecutive cycles.
    Meadow(J,:)    = FlowerField;
    fitMeadow(J,:) = fitFlowers;
    cvMeadow(J)    = cvFlowers;
    if rand<0.50, J=randperm(CSize*N,N); end

    % Global Update for global solutions (archive + leader)
    % The end-of-cycle population is offered to the archive, then the leader
    % for the next cycle is re-drawn from the updated front.
    [Archive, ArchiveFit, ArchiveCV] = updateArchive(Archive, ArchiveFit, ArchiveCV, FlowerField, fitFlowers, cvFlowers, ArchiveSize, ARCHIVE_EPS);
    [gBest, gBestVal] = selectLeader(Archive, ArchiveFit);

    out.archive    = Archive;
    out.archiveFit = ArchiveFit;
    out.archiveCV  = ArchiveCV;
    out.gbest      = gBest;
    out.gval       = gBestVal;
    assignin('base','eBee_MO_out',out);

    % In the multi-objective setting there is no single fMin to report, so the
    % progress indicator is the current front cardinality |PF|.
    fprintf('Iter=%d -->  |PF|=%d \n', iteration, size(ArchiveFit,1));

    % Refresh the wildflowers so the scout phase keeps pointing at genuinely
    % unexplored regions instead of a stale random set.
    if mod(iteration,100)==0, wildflowers = lb + (ub -lb) .* rand(N, D); end
end

end


% ======================= Deb's rule greedy selection ======================
function [X, fitX, cvX, Y, fitY, cvY] = Update(X, fitX, cvX, Y, lb, ub, fObj, userdata, AF, eps0)
% Bound repair + batch evaluation + row-wise Deb's-rule replacement.
%
%   X, fitX, cvX : incumbent population, its objectives and its violations.
%   Y            : trial population (n x D), possibly outside the box.
%   AF, eps0     : current archive objectives and the dominance epsilon; both
%                  are needed by the rule-4 tie-break inside debBetter.
%
% Clamping is saturation (projection onto the nearest bound), not reflection.
% The evaluated trials are also returned unchanged as the last three outputs
% so the caller can offer them to the archive even when they lose here.

Y = max(lb, min(ub, Y));
[fitY, cvY] = evalPop(fObj, Y, userdata);
ind = debBetter(fitY, cvY, fitX, cvX, AF, eps0);
X(ind,:)    = Y(ind,:);
fitX(ind,:) = fitY(ind,:);
cvX(ind)    = cvY(ind);
end


function ind = debBetter(fY, cvY, fX, cvX, AF, eps0)
% Row-wise Deb's rule: true where candidate Y replaces incumbent X.
%
% Precedence: feasibility (rule 1) > violation magnitude (rule 2) >
% epsilon-dominance (rule 3) > archive-lag then sparsity (rule 4).
if nargin < 5, AF = []; end
if nargin < 6, eps0 = 0; end

n   = size(fY,1);
ind = false(n,1);

% Convention: cv <= 0 means feasible, cv > 0 is the total violation amount.
feasY = cvY <= 0;
feasX = cvX <= 0;

% Rule 1: exactly one of the two is feasible
% (the complementary case, X feasible and Y not, leaves ind false = keep X).
ind(feasY & ~feasX) = true;

% Rule 2: both infeasible -> smaller total violation wins
k = ~feasY & ~feasX;
ind(k) = cvY(k) < cvX(k);

% Rules 3-4: both feasible
k = find(feasY & feasX);
if isempty(k), return; end

% Rule-4 material: archive dominance status first, sparsity second.
% Both quantities are computed for the whole subset in one shot, outside the
% per-row loop below, to keep the archive scans vectorized.
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
        % This is improvement (2): a deterministic diversity pressure in place
        % of the v1 coin flip.
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
% An empty archive dominates nothing, so every point is reported as leading.
n = size(F,1);
dom = false(n,1);
if isempty(AF), return; end
for i = 1:n
    a = F(i,:);
    % Standard dominance test broadcast over all archive rows: some member is
    % no worse in every objective and strictly better in at least one.
    dom(i) = any( all(bsxfun(@le, AF - eps0, a), 2) & any(bsxfun(@lt, AF + eps0, a), 2) );
end
end


function [dY, dX] = archiveSparsity(FY, FX, AF)
% Normalised distance from each point to its nearest archive member.
% Larger value = sparser (more valuable) region of objective space.
%
% Objectives are min-max normalised by the archive's own extent so that
% differently scaled objectives contribute comparably to the distance. With
% fewer than two archive members the notion of sparsity is meaningless, so all
% distances are returned as zero, which makes rule 4 fall through to the tie.
nY = size(FY,1);
if isempty(AF) || size(AF,1) < 2
    dY = zeros(nY,1); dX = zeros(nY,1); return;
end
lo  = min(AF,[],1);
hi  = max(AF,[],1);
rng = hi - lo; rng(rng == 0) = 1;   % guard against degenerate (flat) objectives

An = bsxfun(@rdivide, bsxfun(@minus, AF, lo), rng);
Yn = bsxfun(@rdivide, bsxfun(@minus, FY, lo), rng);
Xn = bsxfun(@rdivide, bsxfun(@minus, FX, lo), rng);

dY = nearestDist(Yn, An);
dX = nearestDist(Xn, An);
end


function d = nearestDist(P, A)
% Euclidean distance from every row of P to its nearest row of A.
n = size(P,1);
d = zeros(n,1);
for i = 1:n
    df = bsxfun(@minus, A, P(i,:));
    d(i) = min(sqrt(sum(df.^2, 2)));
end
end


% ============================ archive handling ============================
function [A, AF, ACV] = updateArchive(A, AF, ACV, X, F, CV, maxSize, eps0)
% Offer a batch of solutions (X,F,CV) to the external archive.
%
% Sequence: feasibility filter -> merge -> de-duplicate -> non-dominated
% filter -> crowding-distance truncation.
if nargin < 8, eps0 = 0; end

feas = CV <= 0;
if any(feas)
    % Normal case: only feasible candidates may enter.
    X = X(feas,:); F = F(feas,:); CV = CV(feas);
elseif isempty(AF)
    % No feasible solution found yet: keep the least-violating individuals.
    % This bootstraps the archive on heavily constrained problems, and is
    % overwritten as soon as the first feasible point appears.
    [~, o] = sort(CV); o = o(1:min(maxSize,numel(o)));
    X = X(o,:); F = F(o,:); CV = CV(o);
else
    % A feasible archive already exists and nothing offered is feasible:
    % nothing can improve the front, so return untouched.
    return;
end

A   = [A;   X];
AF  = [AF;  F];
ACV = [ACV; CV];

% Remove duplicate objective vectors ('stable' preserves insertion order, so
% the earliest occurrence and its decision vector survive).
[AF, iu] = unique(AF, 'rows', 'stable');
A = A(iu,:); ACV = ACV(iu);

% Keep only the (epsilon-)non-dominated members of the merged set.
nd = nonDominatedMask(AF, eps0);
A = A(nd,:); AF = AF(nd,:); ACV = ACV(nd);

if size(AF,1) > maxSize
    % Improvement (4): the extreme points are retained unconditionally.
    cd   = crowdingDistance(AF);
    keep = find(isinf(cd));          % boundary points of every objective
    rest = find(~isinf(cd));
    [~, o] = sort(cd(rest), 'descend');   % least crowded survive first
    room = maxSize - numel(keep);
    if room > 0
        keep = [keep; rest(o(1:min(room, numel(rest))))];
    else
        % Degenerate case: more extreme points than the size cap allows.
        keep = keep(1:maxSize);
    end
    keep = sort(keep);
    A = A(keep,:); AF = AF(keep,:); ACV = ACV(keep);
end
end


function [g, gval] = selectLeader(A, AF)
% Binary tournament on crowding distance: the leader comes from the least
% crowded region, which pushes the front outwards instead of inwards.
% The two contestants are drawn with replacement, so they may coincide.
K = size(A,1);
if K == 1, g = A(1,:); gval = AF(1,:); return; end
cd = crowdingDistance(AF);
i1 = randi(K); i2 = randi(K);
if cd(i1) >= cd(i2), i = i1; else, i = i2; end
g = A(i,:); gval = AF(i,:);
end


function nd = nonDominatedMask(F, eps0)
% Non-dominated filter with an early-exit optimisation: a point already marked
% dominated cannot dominate anything that survives, so it is skipped.
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
% NSGA-II crowding distance: per objective, sort the front and accumulate the
% normalised gap between each point's two neighbours. Boundary points receive
% Inf so they are never discarded by truncation.
[K, Mo] = size(F);
cd = zeros(K,1);
if K <= 2, cd = inf(K,1); return; end   % every point is a boundary point
for m = 1:Mo
    [s, o] = sort(F(:,m));
    cd(o(1)) = inf; cd(o(end)) = inf;
    rg = s(end) - s(1);
    if rg == 0, continue; end           % objective is constant: no information
    cd(o(2:end-1)) = cd(o(2:end-1)) + (s(3:end) - s(1:end-2)) / rg;
end
end


function q = scalarQuality(F, CV)
% Scalar quality consistent with Deb's rule, used only to feed the unchanged
% SelectByProbability operator. Lower value = better.
%
% Feasible individuals: q = Pareto rank - 0.5 * normalised crowding distance,
% so rank dominates the ordering while crowding breaks ties within a front,
% and the 0.5 factor keeps a rank-r point always ahead of any rank-(r+1) one.
% Infeasible individuals: q = (worst feasible q + 1) + normalised violation,
% which places them strictly behind every feasible solution.
n = size(F,1);
q = zeros(n,1);
feas = CV <= 0;

if any(feas)
    Ff  = F(feas,:);
    r   = paretoRank(Ff);
    cd  = crowdingDistance(Ff);
    cdn = cd;
    fin = ~isinf(cdn);
    % Inf crowding (boundary points) is replaced by the largest finite value so
    % the arithmetic below stays well defined.
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
% Non-dominated sorting: peel successive fronts off the population, assigning
% rank 1 to the first front, 2 to the next, and so on.
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
% The two-output call is attempted first; if the objective declares only one
% output, the catch branch falls back to the single-output form and treats the
% whole population as feasible. Note that any genuine error raised inside a
% two-output objective would also land in the catch branch, so debug the
% objective with a direct call if unconstrained behaviour is unexpected.
try
    [F, CV] = feval(fObj, X, userdata);
catch
    F  = feval(fObj, X, userdata);
    CV = zeros(size(X,1),1);
end
if isempty(CV), CV = zeros(size(X,1),1); end
CV = CV(:);   % force a column vector regardless of the objective's orientation
end


% ============ eBee internal operators - UNCHANGED from eBee_PureCode ======
function [M, scale] = generateMS(nnectar, nDim, T)
% Produce the perturbation mask M and the Levy step scale.
%
%   T : probability of using a SINGLE shared Levy scale per individual.
%       With probability T, c = 1, so scale is (nnectar x 1) and the same
%       magnitude applies to all dimensions of an individual (coherent,
%       isotropic move). Otherwise c = nDim, giving each dimension its own
%       magnitude (anisotropic, more disruptive move).
%       Large T -> exploitation-oriented (Area/Random-Search).
%       Small T -> exploration-oriented  (Direction-Search).
M = generateMask(nnectar, nDim);
if rand <= T
    c = 1;
else
    c = nDim;
end
% Levy exponent uniform on [1.10, 2.00]: lower beta = heavier tails and more
% frequent long jumps; beta near 2 approaches the Gaussian regime.
beta = 1.10 + rand*0.90;
scale = levy_flight(nnectar, c, beta);
end


function M = generateMask(A, B)
% Binary (A x B) mask controlling which variables are perturbed.
%
% The per-call density k = |randi([0 1]) - rand^randi([2 10])| is bimodal: the
% exponent pushes rand^p toward 1, so k concentrates either near 0 (very
% sparse mask -> coordinate-wise refinement) or near 1 (dense mask -> full
% vector move). h = ceil(k*B) >= 1 guarantees at least one perturbed variable,
% so a trial is never an exact copy of its parent. The density is shared by
% the whole mask while randperm(B) makes the active pattern differ per row.
M=zeros(A,B);
k=abs(randi([0 1],1) - rand.^randi([2 10]) );
for i=1:A
    v=randperm(B);
    h=ceil(k*B);
    M(i,v(1:h))=1;
end
end


function ind=SelectByProbability(x,N)
% Elite-biased index sampling used by the onlooker phase (lower x = better).
%
% The shift y = |x + min(x) + eps| makes the values non-negative, the fifth
% power sharpens the contrast between individuals, and p = f/sum(f) normalises
% them. Selection is nearest-value matching: for each draw a random target
% rand^randi([5 10]) is generated (strongly biased toward 0) and the index
% whose p is closest to that target is returned, which preferentially yields
% small-p, i.e. better, individuals. Sampling is with replacement.
y=abs(x+min(x)+eps);
f=y.^5;
p=f./sum(f);
ind=nan(N,1);
for i=1:N
    [~,ind(i)]=min( abs( p - rand^randi([5 10]) ) );
end
end


function scale = levy_flight(N, ndim, beta)
% Mantegna-style Levy-stable step generator with tail shaping.
%
% U ~ Normal(0, sigma_u^2) and V ~ Normal(0,1) give U/|V|^(1/beta), which is
% asymptotically Levy-distributed with index beta; sigma_u is the standard
% closed-form normalising constant.
num = gamma(1 + beta) * sin(pi * beta / 2);
den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
sigma_u = (num / den)^(1 / beta);

U = sigma_u * randn(N, ndim);
V = randn(N, ndim);

scale = U ./ (abs(V) .^ (1 / beta));

% eBee-specific tail shaping: 10% of the exponents equal 1 (step untouched)
% and 90% are drawn from U(0,3). Exponents below 1 compress the step, above 1
% stretch it, producing a mixture of tiny refinement steps and rare very large
% jumps. ceil() may make the pool slightly longer than N; k(randperm(N)) then
% keeps exactly N shuffled entries.
k = [ones(1,ceil(0.10*N)) 3*rand(1,ceil(0.90*N)) ];
k = k(randperm(N))';
h = sign(scale);
scale = h .* abs(scale) .^ k;

end
