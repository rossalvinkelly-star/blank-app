%% pk_deconvolution.m
%
% Pharmacokinetic deconvolution and ka estimation from oral plasma profiles.
%
% WORKFLOW
%   1. Accept a time × concentration matrix (col 1 = time, remaining = conc).
%      The caller declares time and concentration units; time is converted to
%      hours internally so all PK parameters are reported in h^-1 / L.
%   2. Fit 1-compartment and 2-compartment oral PK models via lsqcurvefit.
%   3. Run Wagner-Nelson deconvolution on EACH replicate column individually
%      to produce per-subject fraction-absorbed profiles shown in Panel B.
%   4. Estimate ka from the slope of ln(1-Fa) vs time on the mean profile.
%   5. Produce a publication-quality three-panel figure.
%
% QUICK START
%   Run this file as-is for two built-in demos (single subject + 6 replicates).
%   Or call the public function:
%
%     results = run_analysis(data_matrix)
%     results = run_analysis(data_matrix, 'dose', 100, 'F', 1.0, ...
%                            'time_unit', 'minutes', 'conc_unit', 'ng/mL', ...
%                            'title', 'My Study')
%
% DATA MATRIX FORMAT
%   Column 1   : time (in the units given by time_unit)
%   Columns 2+ : plasma concentrations (in units given by conc_unit)
%   Multiple concentration columns → each subject analysed individually,
%   mean shown in Panels A & C, all individual Fa profiles shown in Panel B.
%
% SUPPORTED UNITS
%   time_unit : 'hours' (default) | 'minutes' | 'seconds'
%   conc_unit : 'mg/L' (default) | 'ug/mL' | 'ng/mL' | 'nM' | 'uM'
%               conc_unit is cosmetic (axis labels); ensure dose is consistent.
%
% REQUIREMENTS
%   MATLAB R2019b+  (tiledlayout)
%   Optimization Toolbox  (lsqcurvefit, nlparci)
%   exportgraphics needs R2020a+; falls back to print() automatically.
%
% =========================================================================

%% ── Demo 1: single subject | time in hours | conc in ug/mL ──────────────
fprintf('\n%s\n  DEMO 1 — Single subject  |  time: hours  |  conc: ug/mL\n%s\n', ...
        repmat('=',1,62), repmat('=',1,62));

[t1, C_obs1, ~, tp1] = generate_test_data('1comp', 42);

results1 = run_analysis([t1, C_obs1], ...
    'dose',      tp1.D, ...
    'F',         tp1.F, ...
    'time_unit', 'hours', ...
    'conc_unit', 'ug/mL', ...
    'title',     sprintf('Demo 1 — Single Subject  (true k_a = %.2f h^{-1})', tp1.ka), ...
    'save_path', 'pk_demo1_single.png');

%% ── Demo 2: 6 subjects (IIV) | time in MINUTES | conc in ug/mL ──────────
fprintf('\n%s\n  DEMO 2 — 6 subjects (IIV)  |  time: minutes  |  conc: ug/mL\n%s\n', ...
        repmat('=',1,62), repmat('=',1,62));

[data2, pop2] = generate_replicate_data(6, 99);
data2_min     = data2;
data2_min(:,1) = data2_min(:,1) .* 60;   % convert time col: hours → minutes

results2 = run_analysis(data2_min, ...
    'dose',      pop2.D, ...
    'F',         pop2.F, ...
    'time_unit', 'minutes', ...
    'conc_unit', 'ug/mL', ...
    'title',     'Demo 2 — 6 Subjects with IIV  (time in min, k_a pop = 1.50 h^{-1})', ...
    'save_path', 'pk_demo2_replicates.png');

% =========================================================================
% To analyse your own data:
%
%   raw = readmatrix('mydata.csv');   % col 1 = time, col 2+ = conc
%   results = run_analysis(raw, 'dose', 100, 'F', 1.0, ...
%                          'time_unit', 'minutes', 'conc_unit', 'ng/mL');
%
% =========================================================================


%% ════════════════════════════════════════════════════════════════════════
%  PUBLIC FUNCTION: run_analysis
%% ════════════════════════════════════════════════════════════════════════
function results = run_analysis(data_matrix, varargin)
%RUN_ANALYSIS  Full PK deconvolution and ka estimation pipeline.
%
%   results = run_analysis(data_matrix)
%   results = run_analysis(data_matrix, 'dose', D, 'F', F, ...
%                          'time_unit', 'hours', 'conc_unit', 'mg/L', ...
%                          'title', str, 'save_path', path)
%
%   Returns a struct: ka_wn, ka_1cmt, ka_2cmt, popt_*/perr_*,
%                     Fa (mean), Fa_all (per-subject), AUC_inf, figure.

    p = inputParser;
    addRequired(p,  'data_matrix');
    addParameter(p, 'dose',      100.0);
    addParameter(p, 'F',           1.0);
    addParameter(p, 'time_unit', 'hours');
    addParameter(p, 'conc_unit', 'mg/L');
    addParameter(p, 'title',   'PK Deconvolution Analysis');
    addParameter(p, 'save_path', 'pk_deconvolution_analysis.png');
    parse(p, data_matrix, varargin{:});

    dose      = p.Results.dose;
    F         = p.Results.F;
    time_unit = p.Results.time_unit;
    conc_unit = p.Results.conc_unit;
    fig_title = p.Results.title;
    save_path = p.Results.save_path;

    % ── Time conversion to hours ─────────────────────────────────────
    time_scale = get_time_scale(time_unit);
    t     = data_matrix(:,1) .* time_scale;
    C_all = data_matrix(:, 2:end);
    C     = mean(C_all, 2);
    n_sub = size(C_all, 2);

    conc_label = get_conc_label(conc_unit);

    [Cmax, imax] = max(C);

    fprintf('================================================================\n');
    fprintf('  PK DECONVOLUTION & ka ESTIMATION\n');
    fprintf('================================================================\n');
    fprintf('  Time points  : %d\n',   numel(t));
    fprintf('  Subjects     : %d\n',   n_sub);
    fprintf('  Time unit    : %s -> converted to hours\n', time_unit);
    fprintf('  Conc unit    : %s\n',   conc_unit);
    fprintf('  Dose / F     : %.1f / %.2f\n', dose, F);
    fprintf('  Cmax         : %.4f %s  at  t = %.2f h\n', Cmax, conc_unit, t(imax));

    % ── 1-compartment fit ─────────────────────────────────────────────
    fprintf('\n  --- 1-Compartment Model ---\n');
    [popt_1c, perr_1c, ok_1c] = fit_1cmt(t, C, dose, F);

    ka_1c = NaN;  ke_wn = 0.10;
    if ok_1c
        ka_1c = popt_1c(1);  ke_1c = popt_1c(2);  V_1c = popt_1c(3);
        ke_wn = ke_1c;
        fprintf('    ka   = %.4f +/- %.4f h^-1\n', ka_1c, perr_1c(1));
        fprintf('    ke   = %.4f +/- %.4f h^-1\n', ke_1c, perr_1c(2));
        fprintf('    V    = %.4f +/- %.4f L\n',    V_1c,  perr_1c(3));
        fprintf('    t1/2 = %.2f h\n', log(2)/ke_1c);
        fprintf('    Tmax = %.2f h\n', log(ka_1c/ke_1c)/(ka_1c - ke_1c));
        rmse = sqrt(mean((C - one_cmt(t, popt_1c, F, dose)).^2));
        fprintf('    RMSE = %.4f %s\n', rmse, conc_unit);
    end

    % ── 2-compartment fit ─────────────────────────────────────────────
    fprintf('\n  --- 2-Compartment Model ---\n');
    [popt_2c, perr_2c, ok_2c] = fit_2cmt(t, C, dose, F);

    ka_2c = NaN;
    if ok_2c
        ka_2c = popt_2c(1);  k10_2c = popt_2c(2);  V1_2c = popt_2c(5);
        fprintf('    ka  = %.4f +/- %.4f h^-1\n', ka_2c,      perr_2c(1));
        fprintf('    k10 = %.4f +/- %.4f h^-1\n', k10_2c,     perr_2c(2));
        fprintf('    k12 = %.4f +/- %.4f h^-1\n', popt_2c(3), perr_2c(3));
        fprintf('    k21 = %.4f +/- %.4f h^-1\n', popt_2c(4), perr_2c(4));
        fprintf('    V1  = %.4f +/- %.4f L\n',    V1_2c,      perr_2c(5));
        fprintf('    CL  = %.4f L/h\n', k10_2c * V1_2c);
        rmse = sqrt(mean((C - two_cmt(t, popt_2c, F, dose)).^2));
        fprintf('    RMSE = %.4f %s\n', rmse, conc_unit);
    end

    % ── Wagner-Nelson (per-subject) ───────────────────────────────────
    fprintf('\n  --- Wagner-Nelson Deconvolution ---\n');
    fprintf('    ke used : %.4f h^-1  (from 1-CMT fit)\n', ke_wn);

    Fa_all_arr = zeros(numel(t), n_sub);
    auc_infs   = zeros(n_sub, 1);
    for i = 1:n_sub
        [Fa_all_arr(:,i), auc_infs(i)] = wagner_nelson(t, C_all(:,i), ke_wn);
    end
    Fa = mean(Fa_all_arr, 2);

    [ka_wn, t_reg, y_reg] = estimate_ka_wn(t, Fa);
    fprintf('    AUC(0-inf) mean : %.3f h*%s\n', mean(auc_infs), conc_unit);
    fprintf('    ka (W-N)        : %.4f h^-1\n', ka_wn);

    % ── Summary ───────────────────────────────────────────────────────
    fprintf('\n  == ka SUMMARY ==\n');
    if ok_1c,  fprintf('    1-Compartment model : %.4f h^-1\n', ka_1c); end
    if ok_2c,  fprintf('    2-Compartment model : %.4f h^-1\n', ka_2c); end
    fprintf(   '    Wagner-Nelson (W-N) : %.4f h^-1\n', ka_wn);
    fprintf('================================================================\n');

    % ── Build figure ──────────────────────────────────────────────────
    t_fine = linspace(0, t(end), 500)';
    C1_fine = []; C2_fine = [];
    if ok_1c,  C1_fine = one_cmt(t_fine, popt_1c, F, dose); end
    if ok_2c,  C2_fine = two_cmt(t_fine, popt_2c, F, dose); end

    show_rep = n_sub > 1;
    fig = build_figure(t, C, C_all, show_rep, t_fine, C1_fine, C2_fine, ...
                       Fa, Fa_all_arr, show_rep, ...
                       t_reg, y_reg, ka_wn, ka_1c, ka_2c, ...
                       fig_title, conc_label);

    if ~isempty(save_path)
        try
            exportgraphics(fig, save_path, 'Resolution',300, 'BackgroundColor','white');
        catch
            print(fig, save_path, '-dpng', '-r300');
        end
        fprintf('\n  Figure saved -> %s\n', save_path);
    end

    results = struct( ...
        'ka_wn',    ka_wn,   'ka_1cmt',  ka_1c,   'ka_2cmt',  ka_2c, ...
        'popt_1cmt', popt_1c, 'perr_1cmt', perr_1c, ...
        'popt_2cmt', popt_2c, 'perr_2cmt', perr_2c, ...
        'Fa', Fa, 'Fa_all', Fa_all_arr, ...
        'AUC_inf', mean(auc_infs), 'figure', fig);
end


%% ════════════════════════════════════════════════════════════════════════
%  TEST DATA GENERATORS
%% ════════════════════════════════════════════════════════════════════════
function [t, C_obs, C_true, params] = generate_test_data(model, seed)
%GENERATE_TEST_DATA  Single-subject synthetic PK data (8% CV noise).
%   Returns time in hours, concentration in mg/L (= ug/mL).

    if nargin < 1, model = '1comp'; end
    if nargin < 2, seed  = 42;      end
    rng(seed);

    t = [0;0.25;0.5;0.75;1;1.5;2;3;4;6;8;12;16;24];

    if strcmp(model, '1comp')
        params = struct('ka',1.50,'ke',0.15,'V',20.0,'F',1.0,'D',100.0);
        C_true = one_cmt(t, [params.ka,params.ke,params.V], params.F, params.D);
    else
        params = struct('ka',1.50,'k10',0.15,'k12',0.30,'k21',0.10, ...
                        'V1',15.0,'F',1.0,'D',100.0);
        C_true = two_cmt(t, [params.ka,params.k10,params.k12, ...
                              params.k21,params.V1], params.F, params.D);
    end
    C_obs = max(C_true + C_true .* 0.08 .* randn(size(t)), 0);
end


function [data_matrix, pop] = generate_replicate_data(n_subjects, seed)
%GENERATE_REPLICATE_DATA  Multi-subject PK data with ~25% CV IIV.
%   Returns data_matrix [n_timepoints × (1+n_subjects)], time in hours.

    if nargin < 1, n_subjects = 6;  end
    if nargin < 2, seed = 99;       end
    rng(seed);

    t   = [0;0.25;0.5;0.75;1;1.5;2;3;4;6;8;12;16;24];
    pop = struct('ka',1.50,'ke',0.15,'V',20.0,'F',1.0,'D',100.0, ...
                 'omega_ka',0.30,'omega_ke',0.25,'omega_V',0.20);

    C_matrix = zeros(numel(t), n_subjects);
    for i = 1:n_subjects
        ka_i = pop.ka * exp(randn * pop.omega_ka);
        ke_i = pop.ke * exp(randn * pop.omega_ke);
        V_i  = pop.V  * exp(randn * pop.omega_V);
        Ci   = one_cmt(t, [ka_i, ke_i, V_i], pop.F, pop.D);
        C_matrix(:,i) = max(Ci + Ci .* 0.08 .* randn(size(t)), 0);
    end
    data_matrix = [t, C_matrix];
end


%% ════════════════════════════════════════════════════════════════════════
%  PK MODEL FUNCTIONS
%% ════════════════════════════════════════════════════════════════════════
function C = one_cmt(t, params, F, D)
    ka = params(1);  ke = params(2);  V = params(3);
    if abs(ka-ke) < 1e-7, ka = ka + 1e-6; end
    C = max(F.*D.*ka ./ (V.*(ka-ke)) .* (exp(-ke.*t) - exp(-ka.*t)), 0);
end


function C = two_cmt(t, params, F, D)
    ka=params(1); k10=params(2); k12=params(3); k21=params(4); V1=params(5);
    ode = @(~,y)[  -ka.*y(1); ...
                    ka.*y(1) - (k10+k12).*y(2) + k21.*y(3); ...
                    k12.*y(2) - k21.*y(3)];
    [~,Y] = ode45(ode, t, [F*D;0;0], odeset('RelTol',1e-8,'AbsTol',1e-10));
    C = max(Y(:,2)./V1, 0);
end


%% ════════════════════════════════════════════════════════════════════════
%  MODEL FITTING
%% ════════════════════════════════════════════════════════════════════════
function [popt, perr, ok] = fit_1cmt(t, C, D, F)
    ok=false; popt=[]; perr=[];
    fn   = @(p,t) one_cmt(t, p, F, D);
    Cmax = max(C);
    opts = optimoptions('lsqcurvefit','Display','off', ...
                        'MaxFunctionEvaluations',20000,'FunctionTolerance',1e-10);
    try
        [popt,~,res,~,~,~,J] = lsqcurvefit(fn, ...
            [1.0, 0.15, D/(Cmax*5+eps)], t, C, ...
            [0.01,1e-4,0.1], [200,20,5000], opts);
        ci   = nlparci(popt, res, 'Jacobian',J, 'Alpha',0.32);
        perr = (ci(:,2)-ci(:,1))./2;
        ok   = true;
    catch ME
        fprintf('  [1-CMT fit failed: %s]\n', ME.message);
    end
end


function [popt, perr, ok] = fit_2cmt(t, C, D, F)
    ok=false; popt=[]; perr=[];
    fn   = @(p,t) two_cmt(t, p, F, D);
    Cmax = max(C);
    opts = optimoptions('lsqcurvefit','Display','off', ...
                        'MaxFunctionEvaluations',60000,'FunctionTolerance',1e-10);
    try
        [popt,~,res,~,~,~,J] = lsqcurvefit(fn, ...
            [1.0,0.15,0.30,0.10,D/(Cmax*5+eps)], t, C, ...
            [0.01,1e-4,1e-4,1e-4,0.1], [200,20,20,20,5000], opts);
        ci   = nlparci(popt, res, 'Jacobian',J, 'Alpha',0.32);
        perr = (ci(:,2)-ci(:,1))./2;
        ok   = true;
    catch ME
        fprintf('  [2-CMT fit failed: %s]\n', ME.message);
    end
end


%% ════════════════════════════════════════════════════════════════════════
%  WAGNER-NELSON DECONVOLUTION
%% ════════════════════════════════════════════════════════════════════════
function [Fa, auc_inf] = wagner_nelson(t, C, ke)
    n   = numel(t);
    auc = zeros(n,1);
    for i = 2:n
        auc(i) = auc(i-1) + 0.5*(C(i)+C(i-1))*(t(i)-t(i-1));
    end
    auc_inf = auc(end) + C(end)/ke;
    Fa = max(min((C + ke.*auc) ./ (ke*auc_inf + eps), 1.0), 0.0);
end


function [ka_est, t_used, y_used] = estimate_ka_wn(t, Fa)
    mask = Fa < 0.95;
    if sum(mask) < 3, mask = Fa < 0.99;        end
    if sum(mask) < 2, mask = true(size(Fa));   end
    t_used = t(mask);
    y_used = log(max(1 - Fa(mask), 1e-9));
    p      = polyfit(t_used, y_used, 1);
    ka_est = -p(1);
end


%% ════════════════════════════════════════════════════════════════════════
%  UNIT HELPERS
%% ════════════════════════════════════════════════════════════════════════
function s = get_time_scale(unit)
    switch lower(unit)
        case {'hours','hour','h','hr','hrs'},       s = 1.0;
        case {'minutes','minute','min','mins'},     s = 1/60;
        case {'seconds','second','s','sec'},        s = 1/3600;
        otherwise
            warning('Unknown time_unit "%s"; assuming hours.', unit);
            s = 1.0;
    end
end


function lbl = get_conc_label(unit)
    switch lower(unit)
        case 'mg/l',   lbl = 'mg L^{-1}';
        case 'ug/ml',  lbl = '\mug mL^{-1}';
        case 'ng/ml',  lbl = 'ng mL^{-1}';
        case 'nm',     lbl = 'nM';
        case 'um',     lbl = '\muM';
        otherwise,     lbl = unit;
    end
end


%% ════════════════════════════════════════════════════════════════════════
%  PUBLICATION FIGURE
%% ════════════════════════════════════════════════════════════════════════
function fig = build_figure(t, C, C_all, show_conc_rep, ...
                             t_fine, C1, C2, ...
                             Fa, Fa_all, show_fa_rep, ...
                             t_reg, y_reg, ka_wn, ka_1c, ka_2c, ...
                             fig_title, conc_label)

    col_data = [0.172, 0.243, 0.314];
    col_1c   = [0.906, 0.298, 0.235];
    col_2c   = [0.204, 0.596, 0.859];
    col_wn   = [0.153, 0.682, 0.376];

    fig = figure('Units','centimeters','Position',[2 2 42 14.5], ...
                 'Color','white','PaperPositionMode','auto');
    set(fig,'DefaultAxesFontName','Helvetica', ...
            'DefaultTextFontName','Helvetica', ...
            'DefaultAxesFontSize',10, ...
            'DefaultLineLineWidth',1.8);

    tl = tiledlayout(fig,1,3,'TileSpacing','loose','Padding','compact');
    title(tl, fig_title,'FontSize',13,'FontWeight','bold','FontName','Helvetica');

    function style_ax(ax)
        set(ax,'Box','off','TickDir','out','LineWidth',1.1, ...
               'XMinorTick','on','YMinorTick','on');
    end

    % ── Panel A: concentration–time ──────────────────────────────────
    ax1 = nexttile;  hold(ax1,'on');

    if show_conc_rep
        C_sd = std(C_all, 0, 2);
        fill(ax1, [t;flipud(t)], [C+C_sd; flipud(C-C_sd)], col_data, ...
             'FaceAlpha',0.12,'EdgeColor','none');
        for k = 1:size(C_all,2)
            plot(ax1, t, C_all(:,k), 'Color',[col_data,0.22],'LineWidth',0.7);
        end
    end
    if ~isempty(C1)
        plot(ax1,t_fine,C1,'-','Color',col_1c,'LineWidth',2.2, ...
             'DisplayName',sprintf('1-CMT  k_a = %.3f h^{-1}', ka_1c));
    end
    if ~isempty(C2)
        plot(ax1,t_fine,C2,'--','Color',col_2c,'LineWidth',2.2, ...
             'DisplayName',sprintf('2-CMT  k_a = %.3f h^{-1}', ka_2c));
    end
    obs_lbl = 'Observed';
    if show_conc_rep, obs_lbl = 'Observed (mean)'; end
    scatter(ax1,t,C,55,col_data,'filled', ...
            'MarkerEdgeColor','white','LineWidth',0.6,'DisplayName',obs_lbl);

    xlabel(ax1,'Time (h)');
    ylabel(ax1, ['Plasma Concentration (' conc_label ')']);
    th=title(ax1,'A   Plasma Concentration–Time Profile','FontWeight','bold');
    set(th,'HorizontalAlignment','left');
    legend(ax1,'Location','northeast','Box','on','FontSize',8.5);
    xlim(ax1,[0,max(t)*1.03]);  ylim(ax1,[0,max(C)*1.18]);
    style_ax(ax1);

    % ── Panel B: fraction absorbed ────────────────────────────────────
    ax2 = nexttile;  hold(ax2,'on');

    if show_fa_rep
        % Individual Fa curves
        for k = 1:size(Fa_all,2)
            plot(ax2, t, Fa_all(:,k)*100, 'Color',[col_wn,0.28], ...
                 'LineWidth',0.9);
        end
        % ±1 SD band
        Fa_sd = std(Fa_all,0,2);
        fill(ax2, [t;flipud(t)], ...
             [max((Fa-Fa_sd)*100,0); flipud(min((Fa+Fa_sd)*100,100))], ...
             col_wn, 'FaceAlpha',0.22, 'EdgeColor','none', ...
             'DisplayName','Mean \pm 1 SD');
        % Mean profile
        plot(ax2, t, Fa*100, 'o-', 'Color',col_wn, 'LineWidth',2.2, ...
             'MarkerSize',6, 'MarkerFaceColor',col_wn, ...
             'MarkerEdgeColor','white', 'DisplayName','Mean F_a');
        legend(ax2,'Location','southeast','Box','on','FontSize',8.5);
    else
        fill(ax2, [t;flipud(t)], [Fa*100;zeros(numel(Fa),1)], col_wn, ...
             'FaceAlpha',0.15, 'EdgeColor','none');
        plot(ax2, t, Fa*100, 'o-', 'Color',col_wn, 'LineWidth',2.2, ...
             'MarkerSize',6, 'MarkerFaceColor',col_wn, ...
             'MarkerEdgeColor','white', 'DisplayName','Wagner-Nelson');
        legend(ax2,'Location','southeast','Box','on','FontSize',8.5);
    end

    yline(ax2,100,':','Color',[0.5 0.5 0.5],'LineWidth',1.0,'Alpha',0.55);

    ann = {sprintf('k_a (W-N)   = %.3f h^{-1}', ka_wn)};
    if ~isnan(ka_1c), ann{end+1} = sprintf('k_a (1-CMT) = %.3f h^{-1}', ka_1c); end
    if ~isnan(ka_2c), ann{end+1} = sprintf('k_a (2-CMT) = %.3f h^{-1}', ka_2c); end
    text(ax2, 0.97,0.05, strjoin(ann,newline), ...
         'Units','normalized','HorizontalAlignment','right', ...
         'VerticalAlignment','bottom','FontSize',8.5,'FontName','Helvetica', ...
         'BackgroundColor','white','EdgeColor',[0.75 0.75 0.75],'Margin',5);

    xlabel(ax2,'Time (h)');  ylabel(ax2,'Fraction Absorbed (%)');
    th=title(ax2,'B   Fraction Absorbed (Wagner-Nelson)','FontWeight','bold');
    set(th,'HorizontalAlignment','left');
    xlim(ax2,[0,max(t)*1.03]);  ylim(ax2,[0,118]);
    style_ax(ax2);

    % ── Panel C: ln(1-Fa) regression ─────────────────────────────────
    ax3 = nexttile;  hold(ax3,'on');

    valid = Fa < 0.999;
    scatter(ax3, t(valid), log(max(1-Fa(valid),1e-9)), 50, col_data, 'filled', ...
            'MarkerEdgeColor','white','LineWidth',0.6, ...
            'DisplayName','All data points');

    used_mask = ismember(t, t_reg);
    scatter(ax3, t(used_mask), log(max(1-Fa(used_mask),1e-9)), 72, col_wn, 'filled', ...
            'MarkerEdgeColor','white','LineWidth',0.6, ...
            'DisplayName','Absorption-phase data');

    p_fit  = polyfit(t_reg, y_reg, 1);
    t_line = linspace(0, t_reg(end)*1.18, 200)';
    plot(ax3, t_line, polyval(p_fit,t_line), '--', 'Color',col_wn, ...
         'LineWidth',2.2, 'DisplayName', ...
         sprintf('Fit: slope = -%.3f h^{-1}', ka_wn));

    xlabel(ax3,'Time (h)');  ylabel(ax3,'ln(1 - F_a)');
    th=title(ax3,'C   k_a Estimation from ln(1 - F_a)','FontWeight','bold');
    set(th,'HorizontalAlignment','left');
    legend(ax3,'Location','northeast','Box','on','FontSize',8.5);
    xlim(ax3,[0, max(t_reg)*1.25]);
    style_ax(ax3);
end
