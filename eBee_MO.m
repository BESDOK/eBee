%{

eBee_MO : MULTIOBJECTIVE eBee
An Artificial Bee Colony Algorithm with Stochastic Mask-Driven Perturbation
and Elite Guidance for Numerical Optimization : eBee
Besdok, E. 2026, Kayseri, TURKIYE

This is the multi-objective form of eBee, obtained by using Deb's rule
(dominance-based selection). The search machinery of eBee (Employed-Bee,
Onlooker-Bee and Scout-Bee phases, generateMask, generateMS, levy_flight,
SelectByProbability, the Meadow/J handling and the wildflowers refresh) is
IDENTICAL to the original eBee. Only the following points were changed,
because multi-objectivity strictly requires them:

  (1) The fitness matrices are now N x M (M = number of objectives).
  (2) The greedy selection 'fitY < fitX' inside Update() is replaced by
      Deb's rule (dominance-based selection).
  (3) Instead of a single gBest, a non-dominated archive (PF/PS) is kept.
      The gBest used in the direction vectors is a solution drawn at random
      from this archive, so the elite-guidance structure is preserved.
  (4) SelectByProbability expects a scalar quality measure, so a
      'dominated-by count' scalarization (how many solutions dominate a
      given one) is used, which is consistent with Deb's rule. The function
      itself was not modified.

USAGE:
  out = eBee_MO(@zdt1, [], 100, 30,  0, 1, 500);
  plot(out.PF(:,1), out.PF(:,2), '.'); grid on


%{
================================================================================
  eBee_MO - ZDT1 TEST
================================================================================
clear out eBee_MO_out

out = eBee_MO(@zdt1, [], 100, 30, 0, 1, 10000);

f1t    = linspace(0, 1, 500)';
PFtrue = [f1t, 1 - sqrt(f1t)];

figure;
plot(out.PF(:,1), out.PF(:,2), 'r.', 'MarkerSize', 12); hold on
plot(PFtrue(:,1), PFtrue(:,2), 'k-', 'LineWidth', 1.2);
legend('obtained','true', 'Location', 'northeast');
grid on; xlabel('f_1'); ylabel('f_2');
title('eBee\_MO - ZDT1: obtained vs. true Pareto front');

gd  = GD (out.PF, PFtrue);
igd = IGD(out.PF, PFtrue);
hv  = HV2D(out.PF, [1.1, 1.1]);

fprintf('\n=== eBee_MO - ZDT1 Results ===\n');
fprintf('GD  = %.6e\n', gd);
fprintf('IGD = %.6e\n', igd);
fprintf('HV  = %.6f\n', hv);
fprintf('|PF| = %d\n', size(out.PF,1));
%}

%}


function out = eBee_MO(ObjFnc, userdata, N, D, lb, ub, MaxCycle)
CSize=2;

% Setting of Initial Population:
% This process initializes the set of Flowers (i.e., FlowerField) from which honeybees can collect nectar. Technically, FlowerField is an
% [N D]-dimensional matrix. Each row of this matrix corresponds to a D-dimensional Flowers vector.
% That is; FlowerField=[Flowers_i] such that i={1,2,3,...,N}, containing N potential nectar sources (i.e., Flowers).
% Similarly; Flowers_i=[NectarSite_j] such that j={1,2,3,...,D}, containing D NectarSites.

% In this regard, a FlowerField matrix consists of Flowers row-vectors carrying possible nectar sources randomly scattered over a hypothetical Meadow.
% In the eBee algorithm, the concept of Meadow analogically corresponds to the 'search space' of the problem at hand.
Meadow = rand(CSize*N, D) .* (ub-lb) +lb;

% In eBee, the productivity level of each Flowers vector is measured using the objective function (ObjFun) at hand.
% Here, the 'userdata' variable is used to carry additional values that need to be provided to the objective function.
% MO: fitMeadow is now a (CSize*N) x M matrix (M = number of objectives).
fitMeadow = feval(ObjFnc,Meadow,userdata);

% Initialization of Global Solution:
% MO: there is no single 'best' solution. A set (archive) of non-dominated
% solutions is kept instead. For elite guidance, gBest is drawn at random
% from this archive.
ndMask = paretoFrontMO(fitMeadow);
PS     = Meadow(ndMask,:);      % Pareto Set   (decision space)
PF     = fitMeadow(ndMask,:);   % Pareto Front (objective space)
gBest  = PS(randi(size(PS,1)),:);

% The eBee algorithm defines the potential nectar sources over which Scout-Bees can fly across the Meadow
% to discover new nectar sources, using Flowers vectors called 'wildflowers'.
% The primary role of 'wildflowers' vectors is to preserve numerical diversity in the FlowerField matrix.
% The initial forms of 'wildflowers' vectors must be generated before the iterative search process is initiated.
wildflowers = lb + (ub -lb) .* rand(N, D);

%% Main loop: Defines the iterative search process of eBee. This process
% consists of three sub-processes; (1) Employed-Bees, (2) Onlooker-Bees, and (3)
% Scout-Bees based search processes.

J=randperm(CSize*N,N);

for iteration = 1:MaxCycle

    FlowerField=Meadow(J,:);
    fitFlowers=fitMeadow(J,:);

    % The first step of the iterative search process is to store the information belonging to the current FlowerField matrix for later sharing with Onlooker-Bees.
    Flowers0=FlowerField;
    fitFlowers0=fitFlowers;


    %% Employed-Bee Phase: Analogically corresponds to bijective-bio interaction. This phase analogically models the nectar search of hypothetical bees called EmployedBees, flying over the Meadow toward a Flowers source that is necessarily different from their current Flowers vector.
    % EmployedBees search for potential nectar sources by flying over the Meadow along the direction vectors (i.e., evolutionary directions) defined by dx, starting from their current Flowers.
    % During this new nectar source search process, EmployedBees make equal-amplitude flights from the current Flowers vector toward nectar sources lying on the dx direction vectors, without favoring any particular nectar source.
    % The equal-amplitude flight process is controlled by the variable T (T=0.10).
    while 1, j=randperm(N); if sum(j==1:N)==0, break; end, end
    T=0.10; % This facilitates the solution of separable problems where the relationships between variables are loose.
    % In eBee, which nectar sources will be visited is controlled by the binary-valued M matrix. The amplitude of the corresponding flight (i.e., evolutionary step size) is controlled by the 'scale' variable.
    [M,scale] = generateMS(N, D,T);

    if rand < 0.50
        dx = FlowerField(j,:) - FlowerField; % (line-search) Direction vector for the Employed-Bee phase
    else
        % Individual indices from 1 to N (Column vector)
        idx = (1:N)';

        % --- Selection of j1 (From N-1 options different from itself) ---
        j1 = randi(N-1, N, 1);
        j1(j1 >= idx) = j1(j1 >= idx) + 1; % Shift by 1 for those matching their own index (idx)

        % --- Selection of j2 (From N-2 options different from itself and j1) ---
        j2 = randi(N-2, N, 1);
        min_idx = min(idx, j1);
        max_idx = max(idx, j1);

        j2(j2 >= min_idx) = j2(j2 >= min_idx) + 1; % Skip the smaller index
        j2(j2 >= max_idx) = j2(j2 >= max_idx) + 1; % Skip the larger index

        % Vectorized calculation
        w1 = rand(N,1) ;
        w2 = rand(N,1) ;
        k = w1 + w2 ;
        j = k > 1 ;
        w1(j==1) = 1 - w1(j==1); w2(j==1) = 1 - w2(j==1);
        % MO: gBest is an elite solution taken from the non-dominated archive.
        if rand < 0.50, hField=FlowerField; else, hField=gBest ; end
        dx = w1 .* (FlowerField(j1,:)-hField) + w2 .* (FlowerField(j2,:)- hField) ; % area-search
    end

    EmployedBee = FlowerField + M .* scale .* dx;   % Morphogenesis process for the Employed-Bee phase
    % Update #1: The information provided by the Flowers sources (i.e., EmployedBee) obtained in the Employed-Bee phase is injected into FlowerField by applying Deb's rule (dominance-based selection).
    [FlowerField, fitFlowers] = Update(FlowerField, fitFlowers, EmployedBee, lb, ub, ObjFnc, userdata);

    %% Onlooker-Bee Phase: This is a "top-best solutions"-based global search process that favors some nectar sources over others.
    % The purpose of the Onlooker-Bee process is to enable a hypothetical bee to leave its current Flowers and search for nectar sources in Flowers locations containing relatively more productive nectar sources across the Meadow.
    % The Onlooker-Bee phase is a partially elitist search process.
    while 1, j0 = randperm(N); if ( sum(j0 == 1 : N ) == 0 ), break; end, end
    while 1, j1 = randperm(N); if ( sum(j1 == j0 ) == 0 ), break; end, end

    % The T=0.90 setting facilitates the solution of complex problems with hybrid structures among variables.
    T=0.90; % In the Onlooker-Bee phase, some nectar sources can be favored more than others. This process is controlled by the value T=0.90. The T value manages the process of differentiating the amplitudes of flights toward nectar sources.
    [M,scale] = generateMS(N, D, T);

    % In this phase, the index numbers (i.e., ind) of the relatively more productive Flowers vectors are generated using the 'SelectByProbability' function.
    % MO: SelectByProbability expects a scalar quality measure. As the
    % multi-objective quality measure, a 'dominated-by count' scalarization
    % consistent with Deb's rule is used (smaller = better quality).
    qFlowers0 = dominatedCount(fitFlowers0);
    ind=SelectByProbability(qFlowers0,ceil(N/2));

    % To obtain elitist direction vectors, N random selections are made from among the relatively most productive sources, and the indices of the Flowers vectors to fly toward are determined (i.e., indFlowers).
    indFlowers=ind(randi(numel(ind),1,N));

    % Whether the Onlooker-Bee phase will behave in an elitist manner in the current iteration is decided using a random mechanism.
    if rand < rand, j1=indFlowers; end

    % The dx direction vectors used in the Onlooker-Bee Phase are randomly formed in either a bijective (one-to-one projection) or surjective (onto projection) structure.
    % In elitist processes, the Flowers interactions required for dx occur in surjective form. This provides the opportunity to benefit more from a relatively more productive source.
    dx = Flowers0(j1,:) - FlowerField(j0,:);  % Direction vector for the Onlooker-Bee phase
    OnlookerFlowers = FlowerField + M .* scale .* dx;  % Morphogenesis process for the Onlooker-Bee phase

    % Update #2: The Flowers vectors obtained in the Onlooker-Bee phase are injected by applying Deb's rule.
    [nectar1, fitnectar1] = Update(Flowers0, fitFlowers0, OnlookerFlowers, lb, ub, ObjFnc, userdata);

    % Update #3: In the final step of the Onlooker-Bee phase, the FlowerField information is updated.
    % MO: the 'fitnectar1 < fitFlowers' comparison is carried out with Deb's rule.
    j = all(fitnectar1 <= fitFlowers, 2) & any(fitnectar1 < fitFlowers, 2);
    FlowerField(j,:)=nectar1(j,:);
    fitFlowers(j,:)=fitnectar1(j,:);


    %% Scout-Bee Phase: This is the process that enables eBee to avoid the problem of numerical diversity loss in the FlowerField matrix.
    % Analogically, it implies that the bees called ScoutBees are aware of the existence of wildflowers sources they have not directly reached.
    % ScoutBees determine the direction in which they will fly to obtain nectar by utilizing the wildflowers locations. The wildflowers locations are updated every 100 iteration steps. This enables ScoutBee vectors
    % to generate direction vectors without encountering the numerical diversity problem.

    if rand < 0.10
        [M,scale] = generateMS(N, D,0.90);  % In the Scout-Bee phase, nectar sources are favored at different levels with randomly varying amplitudes.
        % This facilitates the solution of strongly-related problems where there are complex relationships among variables.
        dx = wildflowers - FlowerField; % Direction vector for ScoutBee
        ScoutBee = FlowerField + M .* scale .* dx;  % Morphogenesis process for the Scout-Bee phase
        % Update #4: In the final step of the Scout-Bee phase, the FlowerField information is updated by applying Deb's rule.
        [FlowerField, fitFlowers] = Update(FlowerField, fitFlowers, ScoutBee, lb, ub, ObjFnc, userdata);
    end

    % Update Phase #2
    % Update the Meadow sources and fitness values
    Meadow(J,:)=FlowerField;
    fitMeadow(J,:)=fitFlowers;
    % Probability of 50% for continuing to use the currently visited Meadow sources
    if rand<0.50, J=randperm(CSize*N,N); end


    % Update the updating the global solutions.
    % MO: the global solution is the set of non-dominated solutions over the Meadow.
    ndMask = paretoFrontMO(fitMeadow);
    PS     = Meadow(ndMask,:);
    PF     = fitMeadow(ndMask,:);
    gBest  = PS(randi(size(PS,1)),:);   % elite guide: a random solution from the archive

    % Reporting of global solutions to the screen and the MATLAB Workspace
    out.PF = PF;         % Pareto Front (objective space)
    out.PS = PS;         % Pareto Set   (decision space)
    out.gbest = gBest;   % Currently active elite guide Flowers vector
    assignin('base','eBee_MO_out',out); % workspace reporter

    fprintf('%s | Iter=%5.0f -->  |PF|=%4d \n', func2str(ObjFnc), iteration, size(PF,1));

    if mod(iteration,100)==0, wildflowers = lb + (ub -lb) .* rand(N, D); end % saves diversity of evolutionary-direction
end % end of iteration

end % end of function



%%                           SUB-FUNCTIONS

function [X, fitX] = Update(X, fitX, Y, lb, ub, fObj, userdata)
% MO: greedy selection is replaced by Deb's rule (dominance-based selection)
Y = max(lb, min(ub, Y));
fitY = feval(fObj, Y, userdata);
ind = all(fitY <= fitX, 2) & any(fitY < fitX, 2);
X(ind,:) = Y(ind,:);
fitX(ind,:) = fitY(ind,:);
end

function nd = paretoFrontMO(F)
% Logical mask of the Pareto-optimal (non-dominated) set
n = size(F,1);
nd = true(n,1);
for i = 1:n
    if nd(i)
        for j = 1:n
            if i ~= j && all(F(j,:) <= F(i,:)) && any(F(j,:) < F(i,:))
                nd(i) = false;
                break;
            end
        end
    end
end
end

function c = dominatedCount(F)
% Number of solutions that dominate each solution (a scalar quality measure
% consistent with Deb's rule). Smaller value = better quality. It is 0 for
% non-dominated solutions.
n = size(F,1);
c = zeros(n,1);
for i = 1:n
    dom = all(F <= F(i,:), 2) & any(F < F(i,:), 2);
    c(i) = sum(dom);
end
end

function [M, scale] = generateMS(nnectar, nDim, T)
M = generateMask(nnectar, nDim);
if rand <= T  % The T value determines whether the bee collects nectar from all flowers in a local habitat (c=nDim) or from only one flower (c=1)
    c = 1;
else
    c = nDim;
end
beta = 1.10 + rand*0.90;
scale = levy_flight(nnectar, c, beta);
end


function M = generateMask(A, B)
% M: Controls which flowers at the visited nectar source will have nectar collected from them
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

function L = levy_flight(N, ndim, beta)
% Levy Flight generates Levy-distributed random numbers of size N x ndim.

% Calculation of the standard deviation (sigma_u) required for the Mantegna algorithm
num = gamma(1 + beta) * sin(pi * beta / 2);
den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2);
sigma_u = (num / den)^(1 / beta);

% U and V are normally distributed (randn) matrices of size N x M
U = sigma_u * randn(N, ndim);
V = randn(N, ndim);

% Calculation of the Levy step length (Generates both positive and negative values)
L = U ./ (abs(V) .^ (1 / beta));

k = [ones(1,ceil(0.10*N)) 3*rand(1,ceil(0.90*N)) ];
k = k(randperm(N))';
h = sign(L);
L = h .* abs(L) .^ k; % if L==1 Levy-Flight

end