function f = gis_facility_location(P, userdata)
% GIS_FACILITY_LOCATION - GIS-based facility location Pareto problem
% Decision Variables (D = 5), each x layer is normalized in the [0,1] range:
% x1: Land acquisition cost
% x2: Distance to main transportation networks
% x3: Ecological/Environmental sensitivity degree
% x4: Topographic slope (Construction difficulty)
% x5: Distance to existing infrastructure networks (water, power)
%
% Objective 1 (f1): Minimization of Direct Economic Costs
% Objective 2 (f2): Minimization of Environmental Impact and Inaccessibility Costs
% Recommended: D = 5, x in [0,1]

[N, D] = size(P);

% Check for the number of variables (Optional but safe for usage)
if D ~= 5
    warning('This problem strictly requires 5 decision variables (D=5).');
end

% Objective 1: Direct investment cost (Largely determined by land value)
f1 = P(:,1);

% g(x) Function: Spatial penalty score
% Calculates the overall unsuitability (penalty) value of the area via the
% weighted sum of the other 4 GIS variables (transport, environment, slope, infrastructure).
g = 1 + 2 * P(:,2) + 3 * P(:,3) + 1.5 * P(:,4) + 2.5 * P(:,5);

% h(x) Function: Geometry of the Pareto front (Concave trade-off)
% As the economic cost (f1) decreases, the spatial unsuitability (g) tends to increase.
h = 1 - (f1 ./ g).^2;

% Objective 2: Minimization of the Social and Environmental unsuitability score
f2 = g .* h;

% Combining the results
f = [f1, f2];
end