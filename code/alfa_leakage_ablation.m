%% Explainable Fault Detection and Isolation for Fixed-Wing UAVs (ALFA dataset)
% Physics-informed residual features + lightweight ensemble + SHAP consistency
% Evaluation: leave-one-flight-out (LOFO), detection delay, false alarms, isolation
%
% Usage (MATLAB Online):
%   1) Upload and unzip alfa_min.zip so that the folder "alfa_min" is next to this file
%   2) Run this script. Results (CSV, MAT, PNG/PDF figures) are saved in ./results
% Requires: Statistics and Machine Learning Toolbox (shapley: R2021a or newer)

clear; clc; close all; rng(0);

%% ------------------------------------------------------------- configuration
cfg.dataDir = fullfile('..','processed');   % ALFA processed folder (one sub-folder per flight)
cfg.fs      = 10;           % common resampling rate [Hz]
cfg.win     = 3;            % window length [s]
cfg.step    = 0.5;          % window step [s]
cfg.K       = 7;            % persistence: K consecutive fault windows = alarm
cfg.tau     = 0.8;          % alarm if P(fault) >= tau (main operating point)
cfg.prior   = 'uniform';    % class prior of the ensemble
cfg.tauGrid = 0.5:0.05:0.95;% operating-point sweep
cfg.KGrid   = [3 5 7 9];
cfg.nTrees  = 100;          % trees in the bagged ensemble
cfg.maxSplit= 50;           % max splits per tree (keeps the model small)
cfg.nQ      = 15;           % SHAP query windows per fault class
cfg.nBg     = 100;          % SHAP background windows (normal flight)
cfg.outDir  = 'results_v5';
if ~exist(cfg.outDir,'dir'), mkdir(cfg.outDir); end

CL       = ["Normal","Engine","Aileron","Rudder","Elevator"];
faultCls = CL(2:end);
% a priori fault signatures (expected dominant residuals), used for the SHAP check
expected.Engine   = {'e_aspd','e_alt','e_vz'};
expected.Aileron  = {'e_roll','eff_roll_gain','eff_roll_corr'};
expected.Rudder   = {'e_yaw','e_xtrk','eff_yaw_gain','eff_yaw_corr'};
expected.Elevator = {'e_pitch','e_alt','e_vz','eff_pitch_gain','eff_pitch_corr'};

W    = round(cfg.win*cfg.fs);
STEP = round(cfg.step*cfg.fs);

%% ------------------------------------------------------------- load flights
d = dir(cfg.dataDir);
d = d([d.isdir] & ~startsWith({d.name},'.'));
Fc = {};
for i = 1:numel(d)
    name = d(i).name;
    lab  = flightLabel(name);
    if lab == "", continue; end
    fl = loadFlight(fullfile(cfg.dataDir,name), name, cfg.fs);
    fl.label = lab;
    Fc{end+1} = fl; %#ok<SAGROW>
    fprintf('%-62s %-9s %6.1f s  onset %6.1f s\n', name, lab, fl.t(end), fl.onset);
end
F  = [Fc{:}];
nF = numel(F);
fprintf('\nLoaded %d flights.\n', nF);

%% ------------------------------------------------------------- windows and features
Xres = []; Xraw = []; Xcmd = []; Xeff = []; Xthr = []; Xrc = []; Y = strings(0,1); fid = []; Tw = []; fcls = strings(0,1);
for i = 1:nF
    [R, Rw, Cm] = makeSignals(F(i).S);
    [xr, resNames, ends] = winFeat(R,  W, STEP, '');
    [xw, rawNames]       = winFeat(Rw, W, STEP, 'raw_');
    [xc, cmdNames]       = winFeat(Cm, W, STEP, '');
    [xe, effNames]       = effFeat(R, Rw, W, STEP);
    [xt, thrNames]       = winFeat(table(F(i).S.throttle, 'VariableNames', {'leak_thr'}), W, STEP, '');
    [xq, rcNames]        = winFeat(F(i).S(:, {'rc1','rc3','rc4','rc5'}), W, STEP, 'leak_');
    t  = F(i).t(ends);
    y  = repmat("Normal", numel(t), 1);
    y(t >= F(i).onset) = F(i).label;
    Xres = [Xres; xr]; Xraw = [Xraw; xw]; Xcmd = [Xcmd; xc]; Xeff = [Xeff; xe]; Xthr = [Xthr; xt]; Xrc = [Xrc; xq]; %#ok<AGROW>
    Y = [Y; y]; fid = [fid; i*ones(numel(t),1)]; Tw = [Tw; t]; %#ok<AGROW>
    fcls = [fcls; repmat(F(i).label, numel(t), 1)]; %#ok<AGROW>
end
onsets = [F.onset]';
flab   = [F.label]';
fprintf('Windows: %d | features RES %d, RAW %d\n', numel(Y), numel(resNames), numel(rawNames));
disp(groupcounts(categorical(Y)));

sets.names = ["RAW","RES","RES+CMD","RES+CMD+EFF (proposed)","RAW+THR","RAW+RC","RAW+THR+RC","PROPOSED+THR+RC"];
sets.X     = {Xraw, Xres, [Xres Xcmd], [Xres Xcmd Xeff], [Xraw Xthr], [Xraw Xrc], [Xraw Xthr Xrc], [Xres Xcmd Xeff Xthr Xrc]};
iProp      = 4;
propNames  = [resNames cmdNames effNames];

%% ------------------------------------------------------------- LOFO evaluation
Msum = table();
nS = numel(sets.names); Pall = cell(1,nS); PRall = cell(1,nS);
for s = 1:nS
    fprintf('\n=== LOFO: %s ===\n', sets.names(s));
    PR = lofo(sets.X{s}, Y, fid, fcls, cfg, CL);
    P  = decide(PR, cfg.tau, CL);
    Pall{s} = P; PRall{s} = PR;
    M = evalMetrics(P, Y, Tw, fid, fcls, onsets, flab, cfg, CL);
    Msum = [Msum; table(sets.names(s), M.macroF1, M.f1(1), M.f1(2), M.f1(3), M.f1(4), M.f1(5), ...
        M.detRate, M.medDelay, M.meanDelay, M.faN, M.faPerHour, M.isoAcc, ...
        'VariableNames', {'FeatureSet','MacroF1','F1_Normal','F1_Engine','F1_Aileron', ...
        'F1_Rudder','F1_Elevator','DetRate','MedianDelay_s','MeanDelay_s','FA_episodes', ...
        'FA_per_hour','IsolationAcc'})]; %#ok<AGROW>
    Mall(s) = M; %#ok<SAGROW>
end
disp(Msum);
writetable(Msum, fullfile(cfg.outDir,'results_summary.csv'));

% per-flight table for the proposed set
Mp = Mall(iProp);
perFlight = table(string({F.name})', flab, onsets, Mp.delay, Mp.isoPred, ...
    'VariableNames', {'Flight','TrueClass','Onset_s','Delay_s','IsolatedAs'});
writetable(perFlight, fullfile(cfg.outDir,'per_flight_proposed.csv'));

%% ------------------------------------------------------------- paired tests at the fixed operating point
fault = isfinite(onsets) & flab ~= "Multi";
det = zeros(nF, nS);
for s = 1:nS, det(:,s) = ~isnan(Mall(s).delay); end
mc = @(a,b) min(1, 2*binocdf(min(sum(~a&b), sum(a&~b)), sum(~a&b)+sum(a&~b), 0.5));
T2 = table(sets.names', zeros(nS,1), zeros(nS,1), zeros(nS,1), zeros(nS,1), ...
    'VariableNames', {'FeatureSet','DetectedFlights','McNemar_p_vs_RAW','McNemar_p_vs_RES','McNemar_p_vs_proposed'});
for s = 1:nS
    T2.DetectedFlights(s) = sum(det(fault,s));
    T2.McNemar_p_vs_RAW(s) = mc(det(fault,1)==1, det(fault,s)==1);
    T2.McNemar_p_vs_RES(s) = mc(det(fault,2)==1, det(fault,s)==1);
    T2.McNemar_p_vs_proposed(s) = mc(det(fault,iProp)==1, det(fault,s)==1);
end
disp(T2); writetable(T2, fullfile(cfg.outDir,'paired_tests.csv'));
% per-class detected flights per set
cls = ["Engine","Aileron","Rudder","Elevator"]; PC = zeros(nS, 4);
for s = 1:nS, for c = 1:4, PC(s,c) = sum(det(flab==cls(c), s)); end, end
PCt = array2table(PC, 'VariableNames', cellstr(cls)); PCt.FeatureSet = sets.names'; disp(PCt);
writetable(PCt, fullfile(cfg.outDir,'per_class_detected.csv'));

%% ------------------------------------------------------------- window-level split vs LOFO (optimism of window splits)
wsets = [1 iProp 7 8];
q = fcls ~= "Multi"; Wt = table();
for s = wsets
    X = sets.X{s}(q,:); y = Y(q); f = fid(q);
    rng(1); cv = cvpartition(categorical(y), 'KFold', 5);       % random windows, flights mixed
    pw = strings(size(y));
    for k = 1:5
        mdl = trainModel(X(training(cv,k),:), y(training(cv,k)), cfg);
        pw(test(cv,k)) = string(predict(mdl, X(test(cv,k),:)));
    end
    [~, j] = max(PRall{s}(q,:), [], 2); pl = CL(j)';               % LOFO, argmax label
    Wt = [Wt; table(sets.names(s), mean(pw == y), wmacro(pw, y, CL), mean(pl(:) == y), wmacro(pl(:), y, CL), ...
        'VariableNames', {'FeatureSet','WindowAcc_randomSplit','MacroF1_randomSplit','WindowAcc_LOFO','MacroF1_LOFO'})]; %#ok<AGROW>
    fprintf('  window-split check: %s done\n', sets.names(s));
end
disp(Wt); writetable(Wt, fullfile(cfg.outDir,'window_vs_lofo.csv'));
save(fullfile(cfg.outDir,'results_v5.mat'), 'Msum','T2','PCt','Wt','cfg');
zip('v5_final.zip', cfg.outDir); disp('Tayyor: v5_final.zip');

function m = wmacro(p, y, CL)
    f = zeros(1, numel(CL));
    for c = 1:numel(CL)
        tp = sum(p==CL(c) & y==CL(c)); fp = sum(p==CL(c) & y~=CL(c)); fn = sum(p~=CL(c) & y==CL(c));
        f(c) = 2*tp / max(2*tp+fp+fn, 1);
    end
    m = mean(f);
end

%% ===================================================================== functions
function lab = flightLabel(name)
    lab = "";
    if contains(name,'no_ground_truth'), return; end
    if contains(name,'rudder_zero__left_aileron'), lab = "Multi"; return; end
    if contains(name,'no_failure'), lab = "Normal"; return; end
    keys = {'engine','Engine'; 'aileron','Aileron'; 'rudder','Rudder'; 'elevator','Elevator'};
    for k = 1:size(keys,1)
        if contains(name, keys{k,1}), lab = string(keys{k,2}); return; end
    end
end

function fl = loadFlight(folder, name, fs)
    rd  = @(topic) readtable(fullfile(folder,[name '-' topic '.csv']), 'VariableNamingRule','preserve');
    src = cell(0,3);
    ax  = {'roll','pitch','yaw','airspeed'};
    for k = 1:numel(ax)
        T = rd(['mavros-nav_info-' ax{k}]); t = T{:,1}*1e-9;
        src(end+1,:) = {t, T.("field.commanded"), [ax{k} '_c']}; %#ok<AGROW>
        src(end+1,:) = {t, T.("field.measured"),  [ax{k} '_m']}; %#ok<AGROW>
    end
    T = rd('mavros-nav_info-errors'); t = T{:,1}*1e-9;
    src(end+1,:) = {t, T.("field.alt_error"),    'alt_err'};
    src(end+1,:) = {t, T.("field.xtrack_error"), 'xtrk_err'};
    T = rd('mavros-nav_info-velocity'); t = T{:,1}*1e-9;
    for a = 'xyz'
        src(end+1,:) = {t, T.(['field.des_'  a]), ['vdes_'  a]}; %#ok<AGROW>
        src(end+1,:) = {t, T.(['field.meas_' a]), ['vmeas_' a]}; %#ok<AGROW>
    end
    T = rd('mavros-vfr_hud'); t = T{:,1}*1e-9;
    for c = {'throttle','climb','groundspeed'}
        src(end+1,:) = {t, T.(['field.' c{1}]), c{1}}; %#ok<AGROW>
    end
    T = rd('mavros-rc-out'); t = T{:,1}*1e-9;          % servo outputs (leak test only)
    for c = [1 3 4 5]
        src(end+1,:) = {t, T.(sprintf('field.channels%d', c)), sprintf('rc%d', c)}; %#ok<AGROW>
    end
    T = rd('mavros-imu-data'); t = T{:,1}*1e-9;
    for a = 'xyz'
        src(end+1,:) = {t, T.(['field.angular_velocity.'    a]), ['w'   a]}; %#ok<AGROW>
        src(end+1,:) = {t, T.(['field.linear_acceleration.' a]), ['acc' a]}; %#ok<AGROW>
    end
    t0 = max(cellfun(@min, src(:,1)));
    t1 = min(cellfun(@max, src(:,1)));
    tg = (t0:1/fs:t1)';
    S  = table();
    for k = 1:size(src,1)
        [tu, iu] = unique(src{k,1});
        v = double(src{k,2}); v = v(iu);
        S.(src{k,3}) = interp1(tu, v, tg, 'linear');
    end
    S = fillmissing(S, 'nearest');
    onset = inf;
    for ft = {'engines','aileron','rudder','elevator'}
        p = fullfile(folder, [name '-failure_status-' ft{1} '.csv']);
        if isfile(p)
            T = readtable(p, 'VariableNamingRule','preserve');
            if height(T) > 0, onset = min(onset, min(T{:,1})*1e-9 - t0); end
        end
    end
    fl.name = name; fl.t = tg - t0; fl.S = S; fl.onset = onset;
end

function [R, Rw, Cm] = makeSignals(S)
    w = @(a) mod(a + 180, 360) - 180;              % angle wrapping [deg]
    R = table();
    R.e_roll  = w(S.roll_c  - S.roll_m);           % roll tracking residual
    R.e_pitch = w(S.pitch_c - S.pitch_m);          % pitch tracking residual
    R.e_yaw   = w(S.yaw_c   - S.yaw_m);            % heading tracking residual
    R.e_aspd  = S.airspeed_c - S.airspeed_m;       % airspeed tracking residual
    R.e_alt   = S.alt_err;                         % altitude error (autopilot)
    R.e_xtrk  = S.xtrk_err;                        % cross-track error
    R.e_vx    = S.vdes_x - S.vmeas_x;              % velocity residuals
    R.e_vy    = S.vdes_y - S.vmeas_y;
    R.e_vz    = S.vdes_z - S.vmeas_z;
    % NOTE: throttle (vfr_hud) is excluded: it drops to 0 at the injected engine failure (leakage)
    Rw = table(S.roll_m, S.pitch_m, S.airspeed_m, S.groundspeed, S.climb, ...
               S.wx, S.wy, S.wz, S.accx, S.accy, S.accz, 'VariableNames', ...
               {'roll','pitch','aspd','gs','climb','wx','wy','wz','ax','ay','az'});
    Cm = table(S.roll_c, S.pitch_c, S.airspeed_c, 'VariableNames', {'cmd_roll','cmd_pitch','cmd_aspd'});
end

function [X, names] = effFeat(R, Rw, W, STEP)
    % windowed control-effectiveness: least-squares gain and correlation
    % between attitude tracking error and the body rate it should produce
    pairs = {'e_roll','wx','roll'; 'e_pitch','wy','pitch'; 'e_yaw','wz','yaw'};
    n = height(R); ends = (W:STEP:n)'; m = numel(ends);
    X = zeros(m, 2*size(pairs,1)); names = cell(1, 2*size(pairs,1));
    for k = 1:size(pairs,1)
        e = R.(pairs{k,1}); w = Rw.(pairs{k,2});
        for i = 1:m
            ii = ends(i)-W+1:ends(i);
            ee = e(ii) - mean(e(ii)); ww = w(ii) - mean(w(ii));
            X(i,2*k-1) = (ee'*ww) / max(ee'*ee, 1e-6);
            X(i,2*k)   = (ee'*ww) / max(sqrt((ee'*ee)*(ww'*ww)), 1e-9);
        end
        names{2*k-1} = ['eff_' pairs{k,3} '_gain'];
        names{2*k}   = ['eff_' pairs{k,3} '_corr'];
    end
end

function [X, names, ends] = winFeat(T, W, STEP, prefix)
    A = T{:,:}; n = size(A,1); p = size(A,2);
    ends = (W:STEP:n)'; m = numel(ends);
    tt = (0:W-1)' - (W-1)/2; den = tt'*tt;
    X = zeros(m, 5*p);
    for i = 1:m
        B  = A(ends(i)-W+1:ends(i), :);
        mu = mean(B,1);
        X(i,:) = [mu, std(B,0,1), max(abs(B),[],1), (tt'*(B-mu))/den, mean(B(end-4:end,:),1)];
    end
    st = {'mean','std','maxabs','slope','last'}; v = T.Properties.VariableNames;
    names = cell(1,5*p); c = 0;
    for s = 1:5
        for j = 1:p, c = c + 1; names{c} = [prefix v{j} '_' st{s}]; end
    end
end

function mdl = trainModel(X, y, cfg)
    p = size(X,2);
    tmpl = templateTree('MaxNumSplits', cfg.maxSplit, 'NumVariablesToSample', max(1,round(sqrt(p))), ...
                        'Reproducible', true);
    mdl = fitcensemble(X, categorical(y), 'Method','Bag', 'NumLearningCycles', cfg.nTrees, ...
                       'Learners', tmpl, 'Prior', cfg.prior);
end

function PR = lofo(X, Y, fid, fcls, cfg, CL)
    PR = zeros(numel(Y), numel(CL)); nF = max(fid);
    for f = 1:nF
        te = fid == f;
        tr = ~te & fcls ~= "Multi";
        mdl = trainModel(X(tr,:), Y(tr), cfg);
        [~, sc] = predict(mdl, X(te,:));
        [~, loc] = ismember(string(mdl.ClassNames), CL);
        PR(te, loc) = sc;
        if mod(f,10) == 0, fprintf('  fold %d/%d\n', f, nF); end
    end
end

function P = decide(PR, tau, CL)
    % alarm if P(fault) = 1 - P(Normal) >= tau; class = most probable fault
    pf = 1 - PR(:,1);
    [~, j] = max(PR(:,2:end), [], 2);
    P = repmat(CL(1), size(PR,1), 1);
    a = pf >= tau;
    P(a) = CL(1 + j(a));
end

function M = evalMetrics(P, Y, T, fid, fcls, onsets, flab, cfg, CL)
    q = fcls ~= "Multi";
    f1 = zeros(1,numel(CL));
    for c = 1:numel(CL)
        tp = sum(P(q)==CL(c) & Y(q)==CL(c));
        fp = sum(P(q)==CL(c) & Y(q)~=CL(c));
        fn = sum(P(q)~=CL(c) & Y(q)==CL(c));
        f1(c) = 2*tp / max(2*tp + fp + fn, 1);
    end
    nF = max(fid); delay = nan(nF,1); isoPred = strings(nF,1); faN = 0; faT = 0;
    for f = 1:nF
        idx = find(fid == f); [t, o] = sort(T(idx)); p = P(idx(o));
        alarm = double(p ~= "Normal");
        run   = movsum(alarm, [cfg.K-1 0]) >= cfg.K;
        pre   = t < onsets(f);
        faT   = faT + sum(pre)*cfg.step;
        r     = run(pre);
        faN   = faN + sum(diff([0; r(:)]) == 1);
        if isfinite(onsets(f))
            j = find(run & ~pre, 1);
            if isempty(j)
                isoPred(f) = "Missed";
            else
                delay(f) = t(j) - onsets(f);
                v = p(max(1,j-cfg.K+1):j); v = v(v ~= "Normal");   % causal: the K windows ending at the alarm
                if isempty(v), isoPred(f) = "Normal"; else, isoPred(f) = string(mode(categorical(v))); end
            end
        end
    end
    fault = isfinite(onsets) & flab ~= "Multi";
    M.f1 = f1; M.macroF1 = mean(f1);
    M.delay = delay; M.isoPred = isoPred;
    M.detRate   = mean(~isnan(delay(fault)));
    M.medDelay  = median(delay(fault), 'omitnan');
    M.meanDelay = mean(delay(fault), 'omitnan');
    M.faN = faN; M.faPerHour = faN / (faT/3600);
    M.isoAcc = mean(isoPred(fault) == flab(fault));
    M.fault = fault;
end

function phi = classShap(ex, cls, names)
    S  = ex.ShapleyValues;
    vn = S.Properties.VariableNames;
    j  = find(strcmp(vn, char(cls)), 1);
    if isempty(j), error('Class column %s not found in ShapleyValues.', cls); end
    vals = S{:, j};
    pred = string(S.Predictor);
    [~, loc] = ismember(string(names), pred);
    phi = vals(loc);
    phi = phi(:)';
end

function makeFigures(F, Mall, Pall, Y, fcls, Phi, G, faultCls, expected, sets, CL, cfg, iProp, sweep)
    out = cfg.outDir; set(groot, 'defaultAxesFontSize', 9);
    % ---- Fig 1: residual examples (aileron and engine flights)
    ia = find([F.label] == "Aileron", 1); ie = find([F.label] == "Engine", 1);
    fig = figure('Color','w','Units','centimeters','Position',[2 2 17 9]);
    subplot(2,1,1); [R,~,~] = makeSignals(F(ia).S);
    plot(F(ia).t, R.e_roll, 'b'); hold on; xline(F(ia).onset, 'r--', 'LineWidth', 1.2);
    ylabel('e_{roll} [deg]'); title('Aileron failure flight'); grid on;
    subplot(2,1,2); [R,~,~] = makeSignals(F(ie).S);
    plot(F(ie).t, R.e_aspd, 'b'); hold on; xline(F(ie).onset, 'r--', 'LineWidth', 1.2);
    ylabel('e_{V} [m/s]'); xlabel('Time [s]'); title('Engine failure flight'); grid on;
    saveFig(fig, out, 'fig_residuals');
    % mini plots for the architecture figure
    for k = 1:2
        if k == 1, i = ia; [R,~,~] = makeSignals(F(i).S); v = R.e_roll; nm = 'mini_res_aileron';
        else,      i = ie; [R,~,~] = makeSignals(F(i).S); v = R.e_aspd; nm = 'mini_res_engine'; end
        f2 = figure('Color','w','Units','centimeters','Position',[2 2 5 3]);
        plot(F(i).t, v, 'b', 'LineWidth', 0.8); hold on; xline(F(i).onset, 'r', 'LineWidth', 1.2);
        axis tight; set(gca,'XTick',[],'YTick',[]); box on;
        saveFig(f2, out, nm); close(f2);
    end
    % ---- Fig 2: confusion matrix (proposed RES set)
    q = fcls ~= "Multi";
    fig = figure('Color','w','Units','centimeters','Position',[2 2 10 8.5]);
    cc = confusionchart(categorical(Y(q), CL), categorical(Pall{iProp}(q), CL), ...
        'RowSummary','row-normalized', 'Normalization','absolute');
    cc.Title = 'Window-level confusion (LOFO, proposed RES+CMD)';
    saveFig(fig, out, 'fig_confusion');
    % ---- Fig 3: feature-set comparison
    ns = numel(sets.names); vals = zeros(ns,3);
    for s = 1:ns, vals(s,:) = [Mall(s).macroF1, Mall(s).detRate, Mall(s).isoAcc]; end
    fig = figure('Color','w','Units','centimeters','Position',[2 2 14 7]);
    b = bar(categorical(sets.names, sets.names), vals); ylim([0 1.05]); grid on;
    legend({'Window macro-F1','Flight detection rate','Flight isolation accuracy'}, 'Location','southoutside','Orientation','horizontal');
    for k = 1:numel(b), xtips = b(k).XEndPoints; text(xtips, b(k).YEndPoints+0.02, compose('%.2f', b(k).YData), ...
            'HorizontalAlignment','center','FontSize',7); end
    ylabel('Score');
    saveFig(fig, out, 'fig_comparison');
    % ---- Fig 4: detection delay per class (RES)
    M = Mall(iProp); flab = [F.label]'; fault = M.fault & ~isnan(M.delay);
    fig = figure('Color','w','Units','centimeters','Position',[2 2 10 7]);
    boxchart(categorical(flab(fault), faultCls), M.delay(fault)); hold on;
    swarmchart(categorical(flab(fault), faultCls), M.delay(fault), 12, 'k', 'filled', 'MarkerFaceAlpha', 0.5);
    ylabel('Detection delay [s]'); grid on; title('Per-flight detection delay (LOFO)');
    saveFig(fig, out, 'fig_delay');
    % ---- Fig 5: SHAP heatmap with expected signatures
    Pn = Phi ./ sum(Phi, 2);
    fig = figure('Color','w','Units','centimeters','Position',[2 2 16 6.5]);
    lbl = regexprep(G, '_', '-');   % plain labels (avoid TeX subscripts)
    h = heatmap(lbl, cellstr(faultCls), round(Pn, 2)); h.Colormap = flipud(gray(256)) .^ 0.8;
    h.Title = 'Share of mean |SHAP| per residual group';
    h.XLabel = 'Residual signal group'; h.YLabel = 'Fault class';
    saveFig(fig, out, 'fig_shap');
    % mini SHAP bar for the architecture figure (aileron class)
    f2 = figure('Color','w','Units','centimeters','Position',[2 2 5 3.5]);
    ca = find(faultCls == "Aileron"); [~, o] = sort(Pn(ca,:), 'descend'); o = o(1:min(5,end));
    barh(Pn(ca,o), 'FaceColor', [0.3 0.3 0.3]); set(gca,'YTick',1:numel(o),'YTickLabel',lbl(o),'YDir','reverse','FontSize',7);
    box on; saveFig(f2, out, 'mini_shap_aileron'); close(f2);
    % ---- Fig 6: operating-point trade-off (false alarms vs detection delay)
    fig = figure('Color','w','Units','centimeters','Position',[2 2 11 7.5]); hold on;
    Ks = unique(sweep.K)'; mk = {'o','s','^','d'};
    for k = 1:numel(Ks)
        r = sweep(sweep.K == Ks(k), :);
        plot(r.MedianDelay_s, r.FA_per_hour, ['-' mk{k}], 'Color', [0 0 0] + 0.2*(k-1), ...
            'MarkerFaceColor','w', 'DisplayName', sprintf('K = %d', Ks(k)));
    end
    xlabel('Median detection delay [s]'); ylabel('False alarms per hour');
    set(gca,'YScale','log'); grid on; legend('Location','northeast');
    title('Operating points (\tau = 0.50 ... 0.95)');
    saveFig(fig, out, 'fig_tradeoff');
    disp(expected);
end

function saveFig(fig, out, nm)
    drawnow;
    try
        exportgraphics(fig, fullfile(out,[nm '.png']), 'Resolution', 300);
    catch
    end
    d = dir(fullfile(out,[nm '.png']));
    if isempty(d) || d.bytes == 0
        print(fig, fullfile(out,nm), '-dpng', '-r300');
    end
    try
        exportgraphics(fig, fullfile(out,[nm '.pdf']), 'ContentType','vector');
    catch
        print(fig, fullfile(out,nm), '-dpdf', '-bestfit');
    end
end
