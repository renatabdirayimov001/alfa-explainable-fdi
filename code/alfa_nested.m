%% Nested operating-point selection for the ALFA FDI study
% Run AFTER alfa_fdi_v3 in the same MATLAB session (uses its workspace).
% For every outer LOFO test flight f:
%   1) inner 5-fold grouped CV on the remaining flights produces validation posteriors,
%   2) (tau, K) is chosen on these inner posteriors only:
%      maximize detection rate subject to <= faBudget false alarms per hour
%      (ties: smaller median delay; if no point meets the budget: fewest false alarms),
%   3) the chosen (tau, K) is applied to the outer posterior of flight f (PRall from v3).
% The test flight never influences its own threshold.

need = {'sets','Y','fid','fcls','Tw','onsets','flab','cfg','CL','PRall','iProp','F'};
for k = 1:numel(need)
    if ~exist(need{k}, 'var'), error('Run alfa_fdi_v3 first (missing variable %s).', need{k}); end
end

nest.sets     = [1 iProp];        % RAW and proposed (add 2 and 4 for all sets, slower)
nest.inner    = 5;                % inner grouped folds
nest.faBudget = 2;                % false alarms per hour allowed in selection
nest.tauGrid  = 0.5:0.05:0.95;
nest.KGrid    = [3 5 7 9];
o = 'results_nested'; if ~exist(o,'dir'), mkdir(o); end
DetAll = {}; DelayAll = {}; IsoAll = {};

nF = max(fid);
eligible = find(flab ~= "Multi")';           % flights usable for inner training/validation
Summary = table(); Choices = table();
for s = nest.sets
    X = sets.X{s}; PRo = PRall{s};
    fprintf('\n=== Nested selection: %s ===\n', sets.names(s));
    R = struct('det',{},'delay',{},'iso',{},'faN',{},'faT',{},'tau',{},'K',{});
    for f = 1:nF
        trF = setdiff(eligible, f);
        rng(f); g = mod(randperm(numel(trF)), nest.inner) + 1;
        PRin = zeros(numel(Y), numel(CL));
        for k = 1:nest.inner
            va = trF(g == k); tr = trF(g ~= k);
            mtr = ismember(fid, tr); mva = ismember(fid, va);
            mdl = trainModelN(X(mtr,:), Y(mtr), cfg);
            [~, sc] = predict(mdl, X(mva,:));
            [~, loc] = ismember(string(mdl.ClassNames), CL);
            tmp = zeros(sum(mva), numel(CL)); tmp(:, loc) = sc;
            PRin(mva,:) = tmp;
        end
        % grid search on inner validation posteriors
        G = [];
        for tau = nest.tauGrid
            for K = nest.KGrid
                A = aggEval(PRin, Tw, fid, trF, onsets, tau, K, cfg.step, CL);
                G = [G; tau, K, A.det, A.faPerHour, A.medDelay]; %#ok<AGROW>
            end
        end
        ok = G(:,4) <= nest.faBudget;
        if any(ok)
            C = G(ok,:); C(isnan(C(:,5)),5) = inf;
            C = sortrows(C, [-3 5]);
        else
            C = sortrows(G, [4 -3]);
        end
        tauS = C(1,1); KS = C(1,2);
        r = flightEval(PRo(fid==f,:), Tw(fid==f), onsets(f), tauS, KS, cfg.step, CL);
        r.tau = tauS; r.K = KS;
        R(f) = orderfields(r, {'det','delay','iso','faN','faT','tau','K'});
        if mod(f,5) == 0, fprintf('  outer fold %d/%d\n', f, nF); end
    end
    fault = isfinite(onsets) & flab ~= "Multi";
    det   = [R.det]'; delay = [R.delay]'; iso = [R.iso]';
    faN = sum([R.faN]); faT = sum([R.faT]);
    row = table(sets.names(s), mean(det(fault)), mean(iso(fault) == flab(fault)), ...
        median(delay(fault),'omitnan'), faN/(faT/3600), faN, faT/3600, ...
        'VariableNames', {'FeatureSet','DetRate','IsolationAcc','MedianDelay_s','FA_per_hour','FA_episodes','FA_hours'});
    Summary = [Summary; row]; %#ok<AGROW>
    ch = table(repmat(sets.names(s),nF,1), string({F.name})', flab, [R.tau]', [R.K]', det, delay, iso, ...
        'VariableNames', {'FeatureSet','Flight','TrueClass','tau','K','Detected','Delay_s','IsolatedAs'});
    Choices = [Choices; ch]; %#ok<AGROW>
    DetAll{end+1} = det; DelayAll{end+1} = delay; IsoAll{end+1} = iso; %#ok<SAGROW>
    kM = flab == "Multi"; fprintf('  Multi (rudder+aileron) flight: %s\n', strjoin(iso(kM), ', '));
    emr = contains(string({F.name})', 'emr_traj') & flab == "Engine"; ne = flab == "Engine" & ~emr;
    fprintf('  Engine with emergency trajectory: %d/%d | without: %d/%d\n', sum(det(emr)), sum(emr), sum(det(ne)), sum(ne));
    % per-class detection for this set
    for c = ["Engine","Aileron","Rudder","Elevator"]
        k = flab == c;
        fprintf('  %-8s detected %d/%d, isolated %d/%d\n', c, sum(det(k)), sum(k), sum(iso(k)==c), sum(k));
    end
end
% 95% confidence intervals and paired test (flight level)
fault = isfinite(onsets) & flab ~= "Multi"; nFa = sum(fault);
CIlo = zeros(height(Summary),1); CIhi = CIlo; FAlo = CIlo; FAhi = CIlo;
for i = 1:height(Summary)
    x = sum(DetAll{i}(fault)); [~, ci] = binofit(x, nFa); CIlo(i) = ci(1); CIhi(i) = ci(2);
    k = Summary.FA_episodes(i); hrs = Summary.FA_hours(i);
    FAlo(i) = (k > 0) * chi2inv(0.025, 2*k) / 2 / hrs; FAhi(i) = chi2inv(0.975, 2*k + 2) / 2 / hrs;
end
Summary.Det_CI_low = CIlo; Summary.Det_CI_high = CIhi; Summary.FA_CI_low = FAlo; Summary.FA_CI_high = FAhi;
if numel(DetAll) >= 2
    a = DetAll{1}(fault); b = DetAll{end}(fault);
    nb = sum(~a & b); nc = sum(a & ~b);
    pMc = min(1, 2 * binocdf(min(nb, nc), nb + nc, 0.5));
    fprintf('McNemar (exact), proposed vs RAW: %d flights only proposed, %d only RAW, p = %.3f\n', nb, nc, pMc);
    Summary.McNemar_p = [NaN; pMc*ones(height(Summary)-1,1)];
end
disp(Summary);
writetable(Summary, fullfile(o,'nested_summary.csv'));
writetable(Choices, fullfile(o,'nested_per_flight.csv'));
disp('Done.');

%% ----------------------------------------------------------------- functions
function mdl = trainModelN(X, y, cfg)
    p = size(X,2);
    tmpl = templateTree('MaxNumSplits', cfg.maxSplit, 'NumVariablesToSample', max(1,round(sqrt(p))), ...
                        'Reproducible', true);
    mdl = fitcensemble(X, categorical(y), 'Method','Bag', 'NumLearningCycles', cfg.nTrees, ...
                       'Learners', tmpl, 'Prior', cfg.prior);
end

function A = aggEval(PR, Tw, fid, flights, onsets, tau, K, step, CL)
    det = []; dl = []; faN = 0; faT = 0;
    for f = flights
        m = fid == f;
        r = flightEval(PR(m,:), Tw(m), onsets(f), tau, K, step, CL);
        faN = faN + r.faN; faT = faT + r.faT;
        if isfinite(onsets(f)), det(end+1) = r.det; dl(end+1) = r.delay; end %#ok<AGROW>
    end
    A.det = mean(det); A.medDelay = median(dl, 'omitnan'); A.faPerHour = faN / max(faT/3600, eps);
end

function r = flightEval(PR, t, onset, tau, K, step, CL)
    [t, o] = sort(t); PR = PR(o,:);
    pf = 1 - PR(:,1);
    [~, j] = max(PR(:,2:end), [], 2);
    a = pf >= tau;
    p = repmat(CL(1), numel(pf), 1); p(a) = CL(1 + j(a));
    run = movsum(double(a), [K-1 0]) >= K;
    pre = t < onset;
    r.faT = sum(pre)*step; rr = run(pre); r.faN = sum(diff([0; rr(:)]) == 1);
    r.det = false; r.delay = NaN; r.iso = "";
    if isfinite(onset)
        jj = find(run & ~pre, 1);
        if isempty(jj)
            r.iso = "Missed";
        else
            r.det = true; r.delay = t(jj) - onset;
            v = p(max(1, jj-K+1):jj); v = v(v ~= "Normal");   % causal isolation
            if isempty(v), r.iso = "Normal"; else, r.iso = string(mode(categorical(v))); end
        end
    end
end
