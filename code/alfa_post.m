%% Post-processing for the ALFA FDI study (run AFTER alfa_fdi_v3, same workspace)
% Uses the LOFO posteriors already in the workspace (PRall); no retraining.
% Outputs go to ./results_post and are zipped into post_final.zip

o = 'results_post'; if ~exist(o,'dir'), mkdir(o); end
tauOp = 0.75; KOp = 7;                     % reported operating point
tauGrid = 0.5:0.05:0.95; KGrid = [3 5 7 9];
nS = numel(sets.names);

%% 1) operating-point sweep for every feature set
SWall = [];
for s = 1:nS
    for tau = tauGrid
        for K = KGrid
            c2 = cfg; c2.K = K;
            Ms = postMetrics(postDecide(PRall{s}, tau, CL), Y, Tw, fid, fcls, onsets, flab, c2, CL);
            SWall = [SWall; s, tau, K, Ms.detRate, Ms.medDelay, Ms.faPerHour, Ms.isoAcc, Ms.macroF1]; %#ok<AGROW>
        end
    end
end
sweepAll = array2table(SWall, 'VariableNames', {'Set','tau','K','DetRate','MedianDelay_s','FA_per_hour','IsolationAcc','MacroF1'});
sweepAll.SetName = sets.names(sweepAll.Set)';
writetable(sweepAll, fullfile(o,'sweep_all_sets.csv'));

%% 2) all sets at the common operating point
T = table();
for s = 1:nS
    c2 = cfg; c2.K = KOp;
    P  = postDecide(PRall{s}, tauOp, CL);
    M  = postMetrics(P, Y, Tw, fid, fcls, onsets, flab, c2, CL);
    T  = [T; table(sets.names(s), M.macroF1, M.f1(1), M.f1(2), M.f1(3), M.f1(4), M.f1(5), ...
          M.detRate, M.medDelay, M.meanDelay, M.faN, M.faPerHour, M.isoAcc, ...
          'VariableNames', {'FeatureSet','MacroF1','F1_Normal','F1_Engine','F1_Aileron','F1_Rudder', ...
          'F1_Elevator','DetRate','MedianDelay_s','MeanDelay_s','FA_episodes','FA_per_hour','IsolationAcc'})]; %#ok<AGROW>
    Mop(s) = M; Pop{s} = P; %#ok<SAGROW>
end
disp(T); writetable(T, fullfile(o,'results_at_operating_point.csv'));

% per-class flight detection (proposed set)
Mp = Mop(iProp);
cls = ["Engine","Aileron","Rudder","Elevator"];
pc = table();
for c = cls
    k = flab == c;
    pc = [pc; table(c, sum(k), sum(~isnan(Mp.delay(k))), sum(Mp.isoPred(k) == c), median(Mp.delay(k),'omitnan'), ...
        'VariableNames', {'Class','Flights','Detected','CorrectlyIsolated','MedianDelay_s'})]; %#ok<AGROW>
end
disp(pc); writetable(pc, fullfile(o,'per_class_proposed.csv'));
perFlightOp = table(string({F.name})', flab, onsets, Mp.delay, Mp.isoPred, ...
    'VariableNames', {'Flight','TrueClass','Onset_s','Delay_s','IsolatedAs'});
writetable(perFlightOp, fullfile(o,'per_flight_proposed_op.csv'));

%% 3) figures
set(groot, 'defaultAxesFontSize', 9);
% trade-off curves of all sets at K = KOp
f = figure('Color','w','Units','centimeters','Position',[2 2 12 8]); hold on;
mk = {'o','s','^','d'}; ls = {'--',':','-','-.'};
for s = 1:nS
    r = sweepAll(sweepAll.Set == s & sweepAll.K == KOp, :);
    plot(r.FA_per_hour + 0.5, r.DetRate, [ls{s} mk{s}], 'Color', [0 0 0] + 0.22*(s-1)*(s~=iProp), ...
        'LineWidth', 1 + 0.8*(s == iProp), 'MarkerFaceColor', 'w', 'DisplayName', char(sets.names(s)));
end
set(gca, 'XScale', 'log'); grid on; xlabel('False alarms per hour (+0.5, log scale)');
ylabel('Flight-level detection rate'); legend('Location','southeast');
title(sprintf('Detection vs. false alarms (K = %d, \\tau = 0.50...0.95)', KOp));
drawnow; exportgraphics(f, fullfile(o,'fig_tradeoff_sets.png'), 'Resolution', 300);
exportgraphics(f, fullfile(o,'fig_tradeoff_sets.pdf'), 'ContentType', 'vector');

% confusion at the operating point (proposed)
q = fcls ~= "Multi";
f = figure('Color','w','Units','centimeters','Position',[2 2 10 8.5]);
cc = confusionchart(categorical(Y(q), CL), categorical(Pop{iProp}(q), CL), 'RowSummary', 'row-normalized');
cc.Title = sprintf('Window-level confusion (LOFO, \\tau = %.2f)', tauOp);
drawnow; exportgraphics(f, fullfile(o,'fig_confusion_op.png'), 'Resolution', 300);
exportgraphics(f, fullfile(o,'fig_confusion_op.pdf'), 'ContentType', 'vector');

% detection delay per class at the operating point
fault = Mp.fault & ~isnan(Mp.delay);
f = figure('Color','w','Units','centimeters','Position',[2 2 10 7]);
boxchart(categorical(flab(fault), cls), Mp.delay(fault)); hold on;
swarmchart(categorical(flab(fault), cls), Mp.delay(fault), 14, 'k', 'filled', 'MarkerFaceAlpha', 0.5);
ylabel('Detection delay [s]'); grid on; title(sprintf('Per-flight detection delay (\\tau = %.2f, K = %d)', tauOp, KOp));
drawnow; exportgraphics(f, fullfile(o,'fig_delay_op.png'), 'Resolution', 300);
exportgraphics(f, fullfile(o,'fig_delay_op.pdf'), 'ContentType', 'vector');

zip('post_final.zip', o); disp('Tayyor: post_final.zip');

%% ----------------------------------------------------------------- functions
function P = postDecide(PR, tau, CL)
    pf = 1 - PR(:,1);
    [~, j] = max(PR(:,2:end), [], 2);
    P = repmat(CL(1), size(PR,1), 1);
    a = pf >= tau;
    P(a) = CL(1 + j(a));
end

function M = postMetrics(P, Y, T, fid, fcls, onsets, flab, cfg, CL)
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
        run = movsum(double(p ~= "Normal"), [cfg.K-1 0]) >= cfg.K;
        pre = t < onsets(f);
        faT = faT + sum(pre)*cfg.step;
        r = run(pre); faN = faN + sum(diff([0; r(:)]) == 1);
        if isfinite(onsets(f))
            j = find(run & ~pre, 1);
            if isempty(j)
                isoPred(f) = "Missed";
            else
                delay(f) = t(j) - onsets(f);
                v = p(max(1,j-cfg.K+1):min(numel(p), j+10)); v = v(v ~= "Normal");
                if isempty(v), isoPred(f) = "Normal"; else, isoPred(f) = string(mode(categorical(v))); end
            end
        end
    end
    fault = isfinite(onsets) & flab ~= "Multi";
    M.f1 = f1; M.macroF1 = mean(f1); M.delay = delay; M.isoPred = isoPred; M.fault = fault;
    M.detRate = mean(~isnan(delay(fault))); M.medDelay = median(delay(fault),'omitnan');
    M.meanDelay = mean(delay(fault),'omitnan'); M.faN = faN; M.faPerHour = faN/(faT/3600);
    M.isoAcc = mean(isoPred(fault) == flab(fault));
end
