% demo_eBee_MO_gis.m
% Application example: eBee_MO on the GIS Facility Location problem.
% Requires: eBee_MO_english.m, gis_facility_location.m, GD.m, IGD.m, HV2D.m
clc; clear;

% ---------------- problem setup ----------------
ObjFnc   = @gis_facility_location;
userdata = [];
D        = 5;          % GIS variables
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

% ---------------- true Pareto front of GIS Problem ----------------
% The minimum possible value for g(x) is 1 (when x2 through x5 are all 0).
% The function h(x) is 1 - (f1 / g)^2.
% Therefore, the true Pareto optimal front is f2 = 1 - f1^2.
f1t    = linspace(0, 1, 500)';
PFtrue = [f1t, 1 - f1t.^2];

% ---------------- metrics ----------------
gd  = GD(PF, PFtrue);
igd = IGD(PF, PFtrue);
hv  = HV2D(PF, [1.1 1.1]);

fprintf('\n=== eBee_MO - GIS Facility Location Results ===\n');
fprintf('GD   = %e\n', gd);
fprintf('IGD  = %e\n', igd);
fprintf('HV   = %f\n', hv);
fprintf('|PF| = %d\n', size(PF,1));
fprintf('time = %.2f s\n', elapsed);

% ---------------- plot ----------------
figure;
plot(PF(:,1), PF(:,2), 'r.', 'MarkerSize', 12); hold on;
plot(PFtrue(:,1), PFtrue(:,2), 'k-', 'LineWidth', 1.2);
xlabel('f_1 (Economic Cost)'); 
ylabel('f_2 (Environmental & Social Cost)');
title('eBee\_MO - GIS Problem: obtained vs. true Pareto front');
legend('obtained', 'true', 'Location', 'northeast');
grid on; box on;