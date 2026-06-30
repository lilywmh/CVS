% =========================
% Figure 2 — Coefficients
% =========================
features = { ...
    'Verbal agreement (elab.)', 'Question asymmetry', ...
    'Sentiment synchrony', 'Semantic similarity', 'Sent alignment'};
betas = [0.742,  0.767, -0.567, -0.617,  0.001];
pvals = [0.0045, 0.0031, 0.0252, 0.0140, 0.9958];

figure('Units','inches','Position',[1 1 4.8 3.6]); hold on;
b = barh(betas, 0.5, 'FaceColor','flat', 'EdgeColor','none');

colors = zeros(length(betas), 3);
for i = 1:length(betas)
    if pvals(i) < 0.01
        if betas(i) > 0
            colors(i,:) = [0.07 0.40 0.28];   % dark green p < .01
        else
            colors(i,:) = [0.65 0.20 0.20];   % dark red p < .01
        end
    elseif pvals(i) < 0.05
        if betas(i) > 0
            colors(i,:) = [0.10 0.50 0.35];   % green p < .05
        else
            colors(i,:) = [0.75 0.30 0.30];   % red p < .05
        end
    elseif pvals(i) < 0.10
        if betas(i) > 0
            colors(i,:) = [0.60 0.85 0.72];   % light green p < .10
        else
            colors(i,:) = [0.94 0.74 0.71];   % light red p < .10
        end
    else
        colors(i,:) = [0.75 0.75 0.75];       % gray ns
    end
end
b.CData = colors;

xline(0, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8);
set(gca, ...
    'YTick',      1:length(features), ...
    'YTickLabel', features, ...
    'FontSize',   12, ...
    'Box',        'off', ...
    'LineWidth',  0.8, ...
    'YDir',       'reverse');
xlabel('Standardized \beta');
xlim([-1.05 1.25]);

for i = 1:length(betas)
    if pvals(i) < 0.01
        sig = '**';
    elseif pvals(i) < 0.05
        sig = '*';
    elseif pvals(i) < 0.10
        sig = '†';
    else
        sig = '';
    end
    if ~isempty(sig)
        if betas(i) > 0
            x = betas(i) + 0.06;
        else
            x = betas(i) - 0.06;
        end
        text(x, i, sig, 'FontSize', 12, ...
            'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end
end

text(1.2, length(features) + 0.6, ...
    '** p < .01   * p < .05   † p < .10', ...
    'HorizontalAlignment', 'right', 'FontSize', 8);

grid on;
ax = gca;
ax.GridColor = [0.9 0.9 0.9];

exportgraphics(gcf, 'fig2_matlab.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'fig2_matlab.png', 'Resolution', 1200);

% =========================
% Figure 1 — LOO-CV R²
% =========================
models = {'Written', 'Model1', 'Model2'};
loo    = [-0.190, -0.190, 0.417];

figure('Units','inches','Position',[1 1 4.8 3.2]); hold on;
b = bar(loo, 0.5, 'FaceColor', 'flat', 'EdgeColor', 'none');

colors_bar = [...
    0.75 0.30 0.30;
    0.75 0.30 0.30;
    0.10 0.50 0.35];
b.CData = colors_bar;

yline(0, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8);
set(gca, ...
    'XTick',      1:3, ...
    'XTickLabel', models, ...
    'FontSize',   11, ...
    'Box',        'off', ...
    'LineWidth',  0.8);
ylabel('LOO-CV R^2');
ylim([-0.33 0.60]);

for i = 1:length(loo)
    val = loo(i);
    if val >= 0
        text(i, val + 0.02, sprintf('+%.3f', val), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
    else
        text(i, val - 0.02, sprintf('%.3f', val), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
    end
end

title('Model performance (\DeltaR^2 = +.54, F(3,17) = 8.89, p = .001)', 'FontSize', 10);

grid on;
ax = gca;
ax.GridColor = [0.9 0.9 0.9];
ax.GridAlpha = 1;

exportgraphics(gcf, 'fig1_matlab.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'fig1_modelmatlab.png', 'Resolution', 1200);