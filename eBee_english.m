%{

 Erkan Besdok, (2026). "An Elite-Guided Artificial Bee Colony Algorithm with Stochastic Mask-Driven Exploration (eBee)"
 Applied Soft Computing (under review).

%}


function out = eBee_english(ObjFnc, userdata, N, D, lb, ub, MaxCycle)
%
%   out = eBee(ObjFnc, userdata, N, D, lb, ub, MaxCycle)
%
%   eBee minimizes a bound-constrained objective function using a bee-colony
%   metaphor built on three sequential search operators applied per cycle:
%
%       1) Direction-Search (Employed-Bee phase)  : sparse, Levy-scaled moves
%                                                   along differential vectors.
%       2) Area-Search      (Onlooker-Bee phase)  : fitness-biased exploitation
%                                                   around promising flowers.
%       3) Random-Search    (Scout-Bee phase)     : occasional re-injection of
%                                                   diversity from wildflowers.
%
%   A key structural feature is the "Meadow": a population of CSize*N solutions
%   from which only N individuals (the FlowerField) are drawn and evolved in a
%   given cycle. The active subset is periodically re-sampled, so the algorithm
%   maintains an implicit archive that is larger than the working population.
%
%   -------------------------------------------------------------------------
%   INPUTS
%   -------------------------------------------------------------------------
%   ObjFnc    : Objective function handle or function name (string/char).
%               It MUST be vectorized and callable as
%                   fitness = feval(ObjFnc, X, userdata)
%               where X is an (n x D) matrix of n candidate solutions (one row
%               per solution) and fitness is an (n x 1) column vector of
%               objective values. Minimization is assumed (lower is better).
%
%   userdata  : Arbitrary user payload (struct, matrix, cell, ...) forwarded
%               unchanged as the second argument of ObjFnc. Use it to pass
%               problem constants, measured data, penalty weights, etc.
%               Pass [] if the objective needs no extra data.
%
%   N         : Number of active flowers (working population size) evolved in
%               each cycle. The internal Meadow holds CSize*N solutions.
%               Typical values: 20-100. N must be >= 3 because the
%               Direction-Search draws two distinct partners j1, j2 that also
%               differ from the current index.
%
%   D         : Problem dimension, i.e. the number of decision variables.
%
%   lb, ub    : Lower and upper bounds of the search space. Each may be a
%               scalar (applied to all variables) or a (1 x D) row vector of
%               per-variable bounds. They are used both for initialization and
%               for clamping every trial solution back into the feasible box.
%
%   MaxCycle  : Number of iterations (cycles). Each cycle performs at most
%               three objective-function batch evaluations of N solutions:
%               one for Direction-Search, one for Area-Search, and one for
%               Random-Search when it triggers.
%
%   -------------------------------------------------------------------------
%   OUTPUT
%   -------------------------------------------------------------------------
%   out.gbest : (1 x D) best solution vector found over all cycles.
%   out.gval  : Scalar objective value of out.gbest.
%
%   The same structure is also exported to the base workspace as 'eBee_out'
%   at every cycle, so the run can be monitored or interrupted safely.
%
%   -------------------------------------------------------------------------
%   ALGORITHMIC NOTE
%   -------------------------------------------------------------------------
%   eBee is elitist in the archive sense only: gBest/gBestVal never worsen, and
%   every operator uses a greedy per-individual replacement rule (a trial
%   replaces its parent only if it is strictly better). All stochastic
%   decisions use plain rand/randi, so results are reproducible by seeding the
%   generator (e.g. rng(0,'twister')) before calling this function.
%
%   -------------------------------------------------------------------------

% CSize is the "colony size multiplier": the Meadow stores CSize*N solutions
% while only N of them are optimized in any single cycle. CSize = 2 means the
% archive is twice the working population size.
CSize = 2;

% --- Meadow initialization -----------------------------------------------
% Uniform random sampling of CSize*N solutions inside the box [lb, ub].
% Note the ordering: rand .* (ub-lb) + lb maps U(0,1) onto each variable range.
Meadow = rand(CSize*N, D) .* (ub-lb) +lb;

% Batch evaluation of the whole Meadow (one vectorized objective call).
fitMeadow = feval(ObjFnc,Meadow,userdata);

% Initial global best over the entire Meadow.
[fitTheBestFlowers, idx] = min(fitMeadow);
gBest = Meadow(idx, :);          % best position found so far (1 x D)
gBestVal = fitTheBestFlowers;    % its objective value (scalar)

% --- Wildflowers ----------------------------------------------------------
% An auxiliary pool of N purely random, never-evaluated points. It is used only
% by the Scout-Bee (Random-Search) operator as a source of exploration
% directions, and it is refreshed every 100 cycles (see end of the main loop).
wildflowers = lb + (ub -lb) .* rand(N, D);

% --- Active subset selection ---------------------------------------------
% J holds the indices of the N Meadow members that form the current
% FlowerField. randperm(CSize*N, N) guarantees sampling without repetition.
J=randperm(CSize*N,N);

for iteration = 1:MaxCycle

    % ---------------------------------------------------------------------
    % Initialization of the cycle
    % ---------------------------------------------------------------------
    % Copy the active subset out of the Meadow. FlowerField/fitFlowers are the
    % individuals that will be modified during this cycle.
    FlowerField=Meadow(J,:);
    fitFlowers=fitMeadow(J);

    % Frozen snapshot of the cycle-start state. Flowers0/fitFlowers0 are used
    % by the Area-Search as an unchanging reference population, which decouples
    % the onlooker phase from the updates already applied by the employed phase.
    Flowers0=FlowerField;
    fitFlowers0=fitFlowers;

    % ---------------------------------------------------------------------
    % Direction-Search  (Employed-Bee phase)
    % ---------------------------------------------------------------------
    % Build a derangement j of 1:N, i.e. a permutation with no fixed point
    % (j(i) ~= i for every i). This prevents any flower from being paired with
    % itself in the differential term below.
    while 1, j=randperm(N); if sum(j==1:N)==0, break; end, end

    % T is the "single-column probability" passed to generateMS. With T = 0.10
    % the Levy scale is drawn per-dimension in 90% of the cases, producing
    % anisotropic (dimension-wise) step sizes and hence a strongly exploratory
    % employed phase.
    T=0.10;
    [M,scale] = generateMS(N, D, T);

    if rand < 0.50
        % Variant A: simple pairwise differential vector using the derangement.
        dx = FlowerField(j,:) - FlowerField;
    else
        % Variant B: convex two-partner recombination toward a reference field.
        idx = (1:N)';

        % j1: for each row i, a random partner index in 1:N different from i.
        % Drawing in 1:N-1 and shifting whenever j1 >= i yields a uniform
        % distribution over the N-1 admissible values without rejection loops.
        j1 = randi(N-1, N, 1);
        j1(j1 >= idx) = j1(j1 >= idx) + 1;

        % j2: a second partner different from both i and j1. Drawing in 1:N-2
        % and shifting past the smaller then the larger of {i, j1} again gives
        % a uniform draw over the N-2 admissible values.
        j2 = randi(N-2, N, 1);
        min_idx = min(idx, j1);
        max_idx = max(idx, j1);

        j2(j2 >= min_idx) = j2(j2 >= min_idx) + 1;
        j2(j2 >= max_idx) = j2(j2 >= max_idx) + 1;

        % w1, w2 are random weights folded so that w1 + w2 <= 1 (the classic
        % triangle-folding trick). The pair (w1, w2) is therefore uniform on the
        % simplex, keeping the combined step bounded.
        w1 = rand(N,1) ;
        w2 = rand(N,1) ;
        k = w1 + w2 ;
        j = k > 1 ;
        w1(j==1) = 1 - w1(j==1); w2(j==1) = 1 - w2(j==1);

        % hField is the reference the partners are measured against: either the
        % population itself (exploratory, row-wise difference) or the global
        % best gBest (exploitative, all rows pulled toward the incumbent).
        if rand < 0.50, hField=FlowerField; else, hField=gBest ; end
        dx = w1 .* (FlowerField(j1,:)-hField) + w2 .* (FlowerField(j2,:)- hField) ;
    end

    % Trial generation: only the dimensions selected by the binary mask M are
    % perturbed, and the perturbation magnitude is the Levy-distributed scale.
    EmployedBee = FlowerField + M .* scale .* dx;

    % Greedy, per-individual selection (also clamps to [lb, ub]).
    [FlowerField, fitFlowers] = Update(FlowerField, fitFlowers, EmployedBee, lb, ub, ObjFnc, userdata);


    % ---------------------------------------------------------------------
    % Area-Search  (Onlooker-Bee phase)
    % ---------------------------------------------------------------------
    % j0 is a derangement of 1:N (no fixed point), and j1 is a permutation that
    % differs from j0 in every position. Together they define pairs (j1, j0)
    % that are index-wise distinct, avoiding degenerate zero difference vectors.
    while 1, j0 = randperm(N); if ( sum(j0 == 1 : N ) == 0 ), break; end, end
    while 1, j1 = randperm(N); if ( sum(j1 == j0 ) == 0 ), break; end, end

    % T = 0.90: here the Levy scale is a single common value per individual in
    % 90% of the cases, giving more coherent, isotropic (whole-vector) moves,
    % which suits the exploitative character of the onlooker phase.
    T=0.90;
    [M,scale] = generateMS(N, D, T);

    % Fitness-proportional-like selection over the cycle-start fitness values:
    % ceil(N/2) elite-biased indices are drawn, then resampled with replacement
    % to length N. This is the "waggle-dance" recruitment of onlooker bees.
    ind=SelectByProbability(fitFlowers0,ceil(N/2));
    indFlowers=ind(randi(numel(ind),1,N));

    % rand < rand is a fair coin (probability 1/2): half the time the recruited
    % elite indices replace the random permutation j1, biasing the differential
    % vector toward high-quality flowers.
    if rand < rand, j1=indFlowers; end

    % Difference between the frozen snapshot and the (already updated) field.
    dx = Flowers0(j1,:) - FlowerField(j0,:);
    OnlookerFlowers = FlowerField + M .* scale .* dx;

    % The onlooker trials compete against the frozen snapshot Flowers0. This
    % produces an independent candidate set (nectar1) rather than overwriting
    % FlowerField directly.
    [nectar1, fitnectar1] = Update(Flowers0, fitFlowers0, OnlookerFlowers, lb, ub, ObjFnc, userdata);

    % Merge step: a nectar solution enters the FlowerField only where it beats
    % the current flower. Combined with the previous line this realizes a
    % two-front comparison (against the snapshot, then against the live field).
    j=fitnectar1<fitFlowers;
    FlowerField(j,:)=nectar1(j,:);
    fitFlowers(j)=fitnectar1(j);

    % ---------------------------------------------------------------------
    % Random-Search  (Scout-Bee phase)
    % ---------------------------------------------------------------------
    % Triggered with probability 0.10 per cycle. It moves flowers toward the
    % random wildflower pool, which is the algorithm's main stagnation escape
    % mechanism. T = 0.90 again favors a single shared Levy scale per individual.
    if rand < 0.10
        [M,scale] = generateMS(N, D,0.90);
        dx = wildflowers - FlowerField;
        ScoutBee = FlowerField + M .* scale .* dx;
        [FlowerField, fitFlowers] = Update(FlowerField, fitFlowers, ScoutBee, lb, ub, ObjFnc, userdata);
    end

    % ---------------------------------------------------------------------
    % Global Update #1 : write the evolved subset back into the Meadow
    % ---------------------------------------------------------------------
    Meadow(J,:)=FlowerField;
    fitMeadow(J)=fitFlowers;

    % With probability 0.50 a fresh active subset is drawn for the next cycle.
    % Keeping J unchanged the other half of the time lets a promising subset be
    % refined over consecutive cycles instead of being dispersed immediately.
    if rand<0.50, J=randperm(CSize*N,N); end

    % ---------------------------------------------------------------------
    % Global Update for Global solutions (gBest, and gBestVal)
    % ---------------------------------------------------------------------
    % Strict improvement test over the whole Meadow guarantees monotone,
    % non-worsening convergence of the reported best.
    [fBest, idx] = min(fitMeadow);
    if fBest < gBestVal, gBestVal = fBest; gBest = Meadow(idx,:); end

    % Live reporting: the result struct is refreshed and mirrored to the base
    % workspace as 'eBee_out' so that long runs can be inspected or aborted
    % without losing the incumbent solution.
    out.gbest = gBest;
    out.gval = gBestVal;
    assignin('base','eBee_out',out);

fprintf('Iter=%d -->  fMin=%5.16f \n', iteration, gBestVal);

% Wildflowers are regenerated every 100 cycles so that the scout phase keeps
% pointing at genuinely unexplored regions instead of a stale random set.
if mod(iteration,100)==0, wildflowers = lb + (ub -lb) .* rand(N, D); end
end

end


function [X, fitX] = Update(X, fitX, Y, lb, ub, fObj, userdata)
% Update  Bound repair + batch evaluation + greedy elementwise selection.
%
%   [X, fitX] = Update(X, fitX, Y, lb, ub, fObj, userdata)
%
%   X, fitX : current population (n x D) and its objective values (n x 1).
%   Y       : trial population (n x D), possibly outside the box.
%   lb, ub  : bounds; broadcast automatically against Y.
%
%   The clamping strategy is saturation (projection onto the nearest bound),
%   not reflection or re-sampling. Selection is per-row and strictly greedy:
%   row i of X is replaced only if fitY(i) < fitX(i), which makes the operator
%   elitist and monotone.

% Project every trial back into the feasible box [lb, ub].
Y = max(lb, min(ub, Y));

% Single vectorized objective call for the whole trial batch.
fitY = feval(fObj, Y, userdata);

% Strict improvement mask and greedy replacement.
ind = fitY < fitX;
X(ind,:) = Y(ind,:);
fitX(ind) = fitY(ind);
end

function [M, scale] = generateMS(nnectar, nDim, T)
% generateMS  Produce the perturbation mask M and the Levy step scale.
%
%   nnectar : number of individuals (rows).
%   nDim    : problem dimension (columns).
%   T       : probability of using a SINGLE shared Levy scale per individual.
%             With probability T, c = 1, so scale is (nnectar x 1) and the same
%             magnitude applies to all dimensions of an individual (coherent,
%             isotropic move). Otherwise c = nDim, so scale is
%             (nnectar x nDim) and each dimension gets its own magnitude
%             (anisotropic, more disruptive move).
%             Large T  -> exploitation-oriented (used in Area/Random-Search).
%             Small T  -> exploration-oriented  (used in Direction-Search).

M = generateMask(nnectar, nDim);

if rand <= T
    c = 1;
else
    c = nDim;
end

% Levy exponent drawn uniformly from [1.10, 2.00]. Lower beta means heavier
% tails and therefore more frequent long jumps; beta near 2 approaches the
% Gaussian regime.
beta = 1.10 + rand*0.90;

scale = levy_flight(nnectar, c, beta);
end


function M = generateMask(A, B)
% generateMask  Binary (A x B) mask controlling which variables are perturbed.
%
%   The per-call density k is drawn as |randi([0 1]) - rand^randi([2 10])|.
%   The exponent randi([2 10]) pushes rand^p toward 1, so k is bimodal: it
%   concentrates either near 0 (very sparse mask, few variables changed, i.e.
%   coordinate-wise local refinement) or near 1 (dense mask, nearly all
%   variables changed, i.e. full-vector global move).
%
%   h = ceil(k*B) >= 1 guarantees that at least one variable is always
%   perturbed, so a trial is never an exact copy of its parent. For each row a
%   fresh permutation randperm(B) selects WHICH h variables are activated, so
%   the density is common to the whole mask while the pattern differs per row.

M=zeros(A,B);
k=abs(randi([0 1],1) - rand.^randi([2 10]) );
for i=1:A
    v=randperm(B);
    h=ceil(k*B);
    M(i,v(1:h))=1;
end
end


function ind=SelectByProbability(x,N)
% SelectByProbability  Elite-biased index sampling used by the onlooker phase.
%
%   x : (n x 1) fitness vector of the reference population (minimization).
%   N : number of indices to draw.
%
%   The shift y = |x + min(x) + eps| makes the values non-negative, the fifth
%   power f = y.^5 sharpens the contrast between individuals, and p = f/sum(f)
%   normalizes them into a probability-like profile.
%
%   Selection is then performed by nearest-value matching: for each draw, a
%   random target rand^randi([5 10]) is generated and the index whose p is
%   closest to that target is returned. Because rand^p with p in [5,10] is
%   strongly biased toward 0, the routine preferentially returns individuals
%   with small p, which under this transform correspond to the better (lower
%   objective) flowers. Indices may repeat, i.e. sampling is with replacement.

y=abs(x+min(x)+eps);
f=y.^5;
p=f./sum(f);
ind=nan(N,1);
for i=1:N
    [~,ind(i)]=min( abs( p - rand^randi([5 10]) ) );
end
end

function scale = levy_flight(N, ndim, beta)
% levy_flight  Mantegna-style Levy-stable step generator with tail shaping.
%
%   N, ndim : output size (N x ndim).
%   beta    : stability index in (0, 2]; smaller beta -> heavier tails.
%
%   The Mantegna algorithm draws U ~ Normal(0, sigma_u^2) and V ~ Normal(0,1),
%   and forms U / |V|^(1/beta), which is asymptotically Levy-distributed with
%   index beta. sigma_u is the standard closed-form normalizing constant.
%
%   The final block is an additional non-linear tail-shaping step specific to
%   eBee: the exponent vector k contains 10% ones and 90% values drawn from
%   U(0,3), randomly permuted over the individuals. Applying
%   sign(scale) .* |scale|^k leaves ~10% of the steps untouched, compresses
%   steps whose exponent is below 1, and stretches those above 1. The result is
%   a mixture of very small refinement steps and occasional very large jumps,
%   which is what gives the operator its simultaneous local/global behaviour.

num = gamma(1 + beta) * sin(pi * beta / 2);
den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
sigma_u = (num / den)^(1 / beta);

U = sigma_u * randn(N, ndim);
V = randn(N, ndim);

scale = U ./ (abs(V) .^ (1 / beta));

% Per-individual exponent vector: 10% of entries equal 1 (no reshaping),
% 90% drawn from U(0,3). Note ceil() may make the pool slightly longer than N;
% k(randperm(N)) then keeps exactly N shuffled entries.
k = [ones(1,ceil(0.10*N)) 3*rand(1,ceil(0.90*N)) ];
k = k(randperm(N))';

h = sign(scale);
scale = h .* abs(scale) .^ k;

end