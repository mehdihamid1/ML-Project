(() => {
  "use strict";

  const chart = document.getElementById("comparison-chart");
  if (!chart) return;
  const metricButtons = Array.from(document.querySelectorAll("[data-metric]"));
  const sortButtons = Array.from(document.querySelectorAll("[data-sort]"));
  const explanation = document.getElementById("chart-explanation");
  const legend = document.getElementById("chart-legend");
  const tableBody = document.querySelector("#comparison-table tbody");
  const tableCaption = document.querySelector("#comparison-table caption");
  const notice = document.getElementById("dashboard-notice");
  const announcer = document.getElementById("chart-announcer");
  const foldModel = document.getElementById("fold-model");
  const foldMetric = document.getElementById("fold-metric");
  const foldChart = document.getElementById("fold-chart");
  const foldSummary = document.getElementById("fold-summary");
  const foldCaption = document.getElementById("fold-caption");
  const svgNS = "http://www.w3.org/2000/svg";
  const metrics = {
    auc: {key: "auc_mean", deviation: "auc_std", folds: "fold_auc", name: "AUC", scale: 1, decimals: 6, unit: "", spreadUnit: "", deltaUnit: " AUC"},
    accuracy: {key: "accuracy_mean", deviation: "accuracy_std", folds: "fold_accuracy", name: "accuracy", scale: 100, decimals: 3, unit: "%", spreadUnit: " pp", deltaUnit: " percentage points"},
    fit: {key: "fit_seconds_mean", name: "fit time", scale: 1, decimals: 3, unit: " s"},
  };
  const tooltip = element("div", "chart-tooltip");
  tooltip.hidden = true;
  tooltip.setAttribute("aria-hidden", "true");
  document.body.append(tooltip);
  let comparison;
  let metric = "auc";
  let sortKey = "auc_mean";
  let ascending = false;

  function element(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  }

  function svgElement(tag, attributes) {
    const node = document.createElementNS(svgNS, tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  }

  function formatted(value, config, signed = false) {
    return (signed && value > 0 ? "+" : "") + (value * config.scale).toFixed(config.decimals);
  }

  // Round axis ticks (steps of 1, 2 or 5 times a power of ten) that cover the data.
  function niceScale(low, high, target = 5) {
    const rough = (high - low || Math.abs(high) || 1) / target;
    const power = 10 ** Math.floor(Math.log10(rough));
    const step = [1, 2, 5, 10].map(factor => factor * power).find(candidate => candidate >= rough);
    const ticks = [];
    for (let index = Math.floor(low / step + 1e-9); index <= Math.ceil(high / step - 1e-9); index += 1) {
      ticks.push(Number((index * step).toFixed(12)));
    }
    return {ticks, min: ticks[0], max: ticks[ticks.length - 1], decimals: Math.max(0, -Math.floor(Math.log10(step) + 1e-9))};
  }

  function showTooltip(anchor, lines) {
    tooltip.replaceChildren(element("strong", "", lines[0]), ...lines.slice(1).map(line => element("span", "", line)));
    tooltip.hidden = false;
    const box = anchor.getBoundingClientRect();
    const width = tooltip.offsetWidth;
    const height = tooltip.offsetHeight;
    const left = Math.min(Math.max(8, box.left + box.width / 2 - width / 2), window.innerWidth - width - 8);
    const top = box.top - height - 10 >= 8 ? box.top - height - 10 : box.bottom + 10;
    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
  }

  function hideTooltip() {
    tooltip.hidden = true;
  }

  // One tab stop per chart: hover, focus and the arrow keys show the same details.
  function navigable(container, items, describe, anchorOf) {
    let current = -1;
    const clear = () => {
      items.forEach(item => item.classList.remove("active"));
      current = -1;
      hideTooltip();
    };
    const activate = (index, announce) => {
      items.forEach((item, position) => item.classList.toggle("active", position === index));
      current = index;
      const lines = describe(index);
      showTooltip(anchorOf(index), lines);
      if (announce) announcer.textContent = lines.join(". ");
    };
    container.tabIndex = 0;
    container.onfocus = () => activate(current < 0 ? 0 : current, true);
    container.onblur = clear;
    container.onkeydown = event => {
      const last = items.length - 1;
      const next = {ArrowRight: current + 1, ArrowDown: current + 1, ArrowLeft: current - 1, ArrowUp: current - 1, Home: 0, End: last}[event.key];
      if (event.key === "Escape") clear();
      else if (next !== undefined) activate(Math.min(last, Math.max(0, next)), true);
      else return;
      event.preventDefault();
    };
    items.forEach((item, index) => {
      item.onpointerenter = () => activate(index, false);
      item.onpointerleave = clear;
    });
  }

  function orderedModels() {
    return [...comparison.models].sort((a, b) => {
      const delta = a[sortKey] - b[sortKey];
      return (ascending ? delta : -delta) || a.model.localeCompare(b.model);
    });
  }

  function renderComparison() {
    const config = metrics[metric];
    const models = orderedModels();
    const mean = model => model[config.key] * config.scale;
    const spread = model => (config.deviation ? model[config.deviation] * config.scale : 0);
    // Means sit close together, so dots on a fitted axis show the differences honestly;
    // only fit time, a magnitude, uses bars that start at zero.
    const scale = config.deviation
      ? niceScale(Math.min(...models.map(model => mean(model) - spread(model))), Math.max(...models.map(model => mean(model) + spread(model))))
      : niceScale(0, Math.max(...models.map(mean)));
    const percent = value => ((value - scale.min) / (scale.max - scale.min)) * 100 + "%";
    const unit = metric === "auc" ? "" : config.unit;
    const rows = [];
    const anchors = [];
    models.forEach(model => {
      const selected = model.model === comparison.selected_model;
      const row = element("div", "comparison-chart-row" + (selected ? " selected" : ""));
      const label = element("div", "chart-model-label", model.model);
      if (selected) label.append(element("span", "sr-only", ", production model"));
      const plot = element("div", "chart-plot");
      plot.setAttribute("aria-hidden", "true");
      scale.ticks.forEach(tick => {
        const line = element("span", "chart-grid");
        line.style.left = percent(tick);
        plot.append(line);
      });
      const value = element("div", "chart-model-value", formatted(model[config.key], config) + unit);
      let anchor;
      if (config.deviation) {
        const whisker = element("span", "chart-whisker");
        whisker.style.left = percent(mean(model) - spread(model));
        whisker.style.right = "calc(100% - " + percent(mean(model) + spread(model)) + ")";
        anchor = element("span", "chart-dot");
        anchor.style.left = percent(mean(model));
        plot.append(whisker, anchor);
        value.append(element("span", "", "± " + formatted(model[config.deviation], config) + config.spreadUnit));
      } else {
        anchor = element("span", "chart-bar");
        anchor.style.width = percent(mean(model));
        plot.append(anchor);
      }
      row.append(label, plot, value);
      rows.push(row);
      anchors.push(anchor);
    });
    const axis = element("div", "chart-axis");
    axis.setAttribute("aria-hidden", "true");
    scale.ticks.forEach(tick => {
      const label = element("span", "", tick.toFixed(scale.decimals) + unit);
      label.style.left = percent(tick);
      axis.append(label);
    });
    const axisRow = element("div", "chart-axis-row");
    axisRow.append(axis);
    chart.replaceChildren(...rows, axisRow);
    navigable(chart, rows, index => {
      const model = models[index];
      if (!config.deviation) return [formatted(model[config.key], config) + config.unit, model.model + " · mean fit time per fold"];
      const folds = model[config.folds];
      return [
        formatted(model[config.key], config) + unit + " ± " + formatted(model[config.deviation], config) + config.spreadUnit,
        model.model + " · mean " + config.name + " ± fold standard deviation",
        "Folds ranged " + formatted(Math.min(...folds), config) + unit + "–" + formatted(Math.max(...folds), config) + unit,
      ];
    }, index => anchors[index]);
    chart.setAttribute("aria-label", "Cross-validation " + config.name + " comparison. Use the arrow keys to step through the models.");
    legend.classList.toggle("bars", !config.deviation);
    const range = scale.min.toFixed(scale.decimals) + "–" + scale.max.toFixed(scale.decimals) + unit;
    explanation.textContent = config.deviation
      ? "Dots show each model's mean cross-validation " + config.name + (metric === "auc" ? " (from malware probabilities)" : "") +
        "; whiskers show ±1 fold standard deviation. Higher is better. The axis spans " + range + " so that small differences stay visible."
      : "Bars show each model's mean training time per fold, from zero. Lower is faster. Times depend on the machine that ran the experiment.";
    metricButtons.forEach(button => {
      const active = button.dataset.metric === metric;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function renderTableOrder() {
    const rows = new Map(Array.from(tableBody.rows).map(row => [row.dataset.model, row]));
    orderedModels().forEach(model => tableBody.append(rows.get(model.model)));
    const label = sortKey === "auc_mean" ? "mean cross-validation AUC" : sortKey === "accuracy_mean" ? "mean cross-validation accuracy" : "mean fit time";
    tableCaption.textContent = "Best recorded configuration of each model, sorted by " + label + ", " + (ascending ? "ascending" : "descending") + ". AUC ranks preserve the original selection order. Standard deviation is across folds, not a confidence interval.";
    sortButtons.forEach(button => {
      const active = button.dataset.sort === sortKey;
      button.closest("th").setAttribute("aria-sort", active ? (ascending ? "ascending" : "descending") : "none");
      button.querySelector(".sort-indicator").textContent = active ? (ascending ? "↑" : "↓") : "";
    });
  }

  function roundedBar(x, y, width, height, roundTop) {
    const radius = Math.min(4, height, width / 2);
    return roundTop
      ? `M${x},${y + height}V${y + radius}Q${x},${y} ${x + radius},${y}H${x + width - radius}Q${x + width},${y} ${x + width},${y + radius}V${y + height}Z`
      : `M${x},${y}H${x + width}V${y + height - radius}Q${x + width},${y + height} ${x + width - radius},${y + height}H${x + radius}Q${x},${y + height} ${x},${y + height - radius}Z`;
  }

  function renderFolds() {
    if (!foldChart) return;
    const config = metrics[foldMetric.value];
    const production = comparison.models.find(model => model.model === comparison.selected_model);
    const other = comparison.models.find(model => model.model === foldModel.value);
    const pairs = production[config.folds].map((value, index) => ({fold: index + 1, production: value, other: other[config.folds][index], difference: value - other[config.folds][index]}));
    const differences = pairs.map(pair => pair.difference * config.scale);
    const scale = niceScale(Math.min(0, ...differences), Math.max(0, ...differences), 4);
    const width = Math.max(300, foldChart.clientWidth);
    const height = 230;
    const margin = {left: 66, right: 8, top: 10, bottom: 28};
    const band = (width - margin.left - margin.right) / pairs.length;
    const barWidth = Math.min(24, band * 0.56);
    const y = value => margin.top + (scale.max - value) / (scale.max - scale.min) * (height - margin.top - margin.bottom);
    const tickLabel = tick => (tick > 0 ? "+" : "") + (tick === 0 ? "0" : tick.toFixed(scale.decimals));
    const svg = svgElement("svg", {viewBox: `0 0 ${width} ${height}`, width, height, "aria-hidden": "true"});
    scale.ticks.forEach(tick => {
      svg.append(svgElement("line", {class: tick === 0 ? "fold-zero" : "fold-grid", x1: margin.left, x2: width - margin.right, y1: y(tick), y2: y(tick)}));
      const label = svgElement("text", {class: "fold-tick", x: margin.left - 10, y: y(tick) + 4, "text-anchor": "end"});
      label.textContent = tickLabel(tick);
      svg.append(label);
    });
    const hits = [];
    const bars = [];
    pairs.forEach((pair, index) => {
      const left = margin.left + band * index;
      const value = pair.difference * config.scale;
      const hit = svgElement("rect", {class: "fold-hit", x: left, y: margin.top, width: band, height: height - margin.top - margin.bottom});
      const top = y(Math.max(0, value));
      const bar = svgElement("path", {class: "fold-bar " + (value >= 0 ? "production" : "other"), d: roundedBar(left + (band - barWidth) / 2, top, barWidth, Math.max(1, y(Math.min(0, value)) - top), value >= 0)});
      const label = svgElement("text", {class: "fold-tick", x: left + band / 2, y: height - 8, "text-anchor": "middle"});
      label.textContent = String(pair.fold);
      svg.append(hit, bar, label);
      hits.push(hit);
      bars.push(bar);
    });
    foldChart.replaceChildren(svg);
    const unit = config === metrics.accuracy ? config.unit : "";
    navigable(foldChart, hits, index => {
      const pair = pairs[index];
      return [
        formatted(pair.difference, config, true) + config.deltaUnit,
        "Fold " + pair.fold + " · " + production.model + " minus " + other.model,
        production.model + " " + formatted(pair.production, config) + unit + " · " + other.model + " " + formatted(pair.other, config) + unit,
      ];
    }, index => bars[index]);

    const higher = pairs.filter(pair => pair.difference > 0).length;
    const lower = pairs.filter(pair => pair.difference < 0).length;
    const ties = pairs.length - higher - lower;
    const meanDifference = pairs.reduce((total, pair) => total + pair.difference, 0) / pairs.length;
    foldSummary.replaceChildren(
      production.model + "'s " + config.name + " was higher in ", element("strong", "", higher + " of " + pairs.length), " folds" +
      (lower ? "; " + other.model + " was higher in " + lower : "") + (ties ? "; " + ties + " tied" : "") + ". Mean difference: ",
      element("strong", "", formatted(meanDifference, config, true)), config.deltaUnit + ".",
    );
    document.getElementById("fold-legend-other").textContent = other.model + " higher";
    document.getElementById("fold-head-other").textContent = other.model;
    document.getElementById("fold-table-caption").textContent = config.name[0].toUpperCase() + config.name.slice(1) + " on each validation fold";
    foldCaption.textContent = "Bars show " + production.model + " minus " + other.model + " on each fold, in " + (config === metrics.auc ? "AUC" : "percentage points") +
      "; above zero means " + production.model + " scored higher. This describes the recorded folds; it is not a significance test.";
    document.querySelector("#fold-table tbody").replaceChildren(...pairs.map(pair => {
      const row = element("tr");
      const fold = element("th", "", String(pair.fold));
      fold.scope = "row";
      row.append(fold, element("td", "", formatted(pair.production, config) + unit), element("td", "", formatted(pair.other, config) + unit),
        element("td", "", formatted(pair.difference, config, true) + (config === metrics.accuracy ? " pp" : "")));
      return row;
    }));
  }

  async function initialize() {
    try {
      const response = await fetch("/api/model-comparison", {headers: {Accept: "application/json"}, credentials: "same-origin"});
      if (!response.ok) throw new Error("Could not load comparison");
      comparison = await response.json();
      renderComparison();
      renderFolds();
      metricButtons.forEach(button => {
        button.disabled = false;
        button.addEventListener("click", () => {
          metric = button.dataset.metric;
          renderComparison();
        });
      });
      sortButtons.forEach(button => {
        button.disabled = false;
        button.addEventListener("click", () => {
          if (sortKey === button.dataset.sort) ascending = !ascending;
          else {
            sortKey = button.dataset.sort;
            ascending = sortKey === "fit_seconds_mean";
          }
          renderTableOrder();
          renderComparison();
          notice.textContent = "Sorted by " + (sortKey === "auc_mean" ? "AUC" : sortKey === "accuracy_mean" ? "accuracy" : "fit time") + ", " + (ascending ? "ascending" : "descending") + ". AUC ranks remain the original selection ranks.";
        });
      });
      if (foldChart) {
        foldModel.disabled = false;
        foldMetric.disabled = false;
        foldModel.addEventListener("change", renderFolds);
        foldMetric.addEventListener("change", renderFolds);
        let resizeFrame;
        window.addEventListener("resize", () => {
          cancelAnimationFrame(resizeFrame);
          resizeFrame = requestAnimationFrame(renderFolds);
        });
      }
      window.addEventListener("scroll", hideTooltip, {passive: true});
    } catch (error) {
      notice.textContent = "Interactive charts could not be loaded. Every recorded value is in the tables on this page.";
    }
  }
  initialize();
})();
