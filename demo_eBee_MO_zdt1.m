% demo_eBee_MO_zdt1.m
% Application example: eBee_MO (Deb's rule) on the ZDT1 benchmark.
% Requires: eBee_MO.m, zdt1.m, GD.m, IGD.m, HV2D.m
clc; clear;

% ---------------- problem setup ----------------
ObjFnc   = @zdt1;
userdata = [];
D        = 30;
lb       = zeros(1,D);
ub       = ones(1,D);

% ---------------- algorithm setup ----------------
N           = 100;     % colony size
MaxCycle    = 1000;    % iterations (~2.1*N evaluations per cycle)
ArchiveSize = 200;     % external Pareto archive capacity

rng('shuffle');
tic;
out = eBee_MO_english(ObjFnc, userdata, N, D, lb, ub, MaxCycle, ArchiveSize);
elapsed = toc;

PF = out.archiveFit;                 % approximated Pareto front (K x 2)
PF = sortrows(PF, 1);

% ---------------- true Pareto front of ZDT1 ----------------
f1t    = linspace(0, 1, 500)';
PFtrue = [f1t, 1 - sqrt(f1t)];

% ---------------- metrics ----------------
gd  = GD(PF, PFtrue);
igd = IGD(PF, PFtrue);
hv  = HV2D(PF, [1.1 1.1]);

fprintf('\n=== eBee_MO - ZDT1 Results ===\n');
fprintf('GD   = %e\n', gd);
fprintf('IGD  = %e\n', igd);
fprintf('HV   = %f\n', hv);
fprintf('|PF| = %d\n', size(PF,1));
fprintf('time = %.2f s\n', elapsed);

% ---------------- plot ----------------
figure;
plot(PF(:,1), PF(:,2), 'r.', 'MarkerSize', 12); hold on;
plot(PFtrue(:,1), PFtrue(:,2), 'k-', 'LineWidth', 1.2);
xlabel('f_1'); ylabel('f_2');
title('eBee\_MO - ZDT1: obtained vs. true Pareto front');
legend('obtained', 'true', 'Location', 'northeast');
grid on; box on;
