const REPO_URL = "https://github.com/arshmann2003/HeartDiseaseClassifier";

const state = {
  artifact: null,
  values: {},
  sessionId: null,
  ref: "direct",
  editedSent: false,
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  bindTabs();
  const form = document.getElementById("profile-form");
  form.addEventListener("submit", (event) => event.preventDefault());
  form.addEventListener("input", onFormInput);
  form.addEventListener("click", onFormClick);

  let artifact;
  try {
    const response = await fetch("model.json");
    if (!response.ok) throw new Error(`model.json returned ${response.status}`);
    artifact = await response.json();
  } catch (error) {
    document.getElementById("lede").textContent =
      "The model file did not load. Start a local server in the built site (python scripts/build_site.py, then python -m http.server -d _site) instead of opening this file directly.";
    return;
  }

  state.artifact = artifact;
  for (const feature of artifact.features) state.values[feature.name] = feature.default;
  renderLede();
  renderPresets();
  renderForm();
  renderResultsShell();
  renderPerformance();
  renderAbout();
  startAnalytics();
  paint();
}

function bindTabs() {
  const tabs = [...document.querySelectorAll("[role='tab']")];
  for (const tab of tabs) {
    tab.addEventListener("click", () => selectTab(tab, tabs));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      const offset = event.key === "ArrowRight" ? 1 : -1;
      const next = tabs[(tabs.indexOf(tab) + offset + tabs.length) % tabs.length];
      next.focus();
      selectTab(next, tabs);
    });
  }
}

function selectTab(tab, tabs) {
  for (const item of tabs) {
    const selected = item === tab;
    item.setAttribute("aria-selected", selected ? "true" : "false");
    document.getElementById(item.getAttribute("aria-controls")).hidden = !selected;
  }
}

function renderLede() {
  const artifact = state.artifact;
  const lede = document.getElementById("lede");
  lede.replaceChildren();
  lede.append(
    document.createTextNode(
      `A logistic regression trained on ${artifact.n_rows.toLocaleString("en-US")} CDC health survey responses estimates the chance that someone with a given profile has heart disease. It runs in this page, so opening the link does not wake a server. `,
    ),
  );
  const link = document.createElement("a");
  link.href = REPO_URL;
  link.textContent = "View the code on GitHub";
  lede.append(link, document.createTextNode("."));
}

function renderPresets() {
  const root = document.getElementById("presets");
  state.artifact.presets.forEach((preset, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset";
    button.dataset.preset = String(index);
    button.textContent = preset.name;
    button.addEventListener("click", () => applyPreset(preset));
    root.append(button);
  });
}

function applyPreset(preset) {
  state.values = { ...preset.values };
  renderForm();
  paint();
  track("preset_clicked", { preset: preset.name });
}

function renderForm() {
  const form = document.getElementById("profile-form");
  form.replaceChildren();
  for (const group of state.artifact.groups) {
    const section = document.createElement("section");
    section.className = "group";
    const heading = document.createElement("h2");
    heading.textContent = group;
    section.append(heading);
    for (const feature of state.artifact.features) {
      if (feature.group === group) section.append(renderField(feature));
    }
    form.append(section);
  }
}

function renderField(feature) {
  const wrap = document.createElement("div");
  wrap.className = "field";
  if (feature.control === "toggle") wrap.append(renderToggle(feature));
  else if (feature.control === "radio") wrap.append(renderRadio(feature));
  else if (feature.control === "select") wrap.append(renderSelect(feature));
  else wrap.append(renderRange(feature));
  if (feature.help) {
    const help = document.createElement("p");
    help.className = "help";
    help.textContent = feature.help;
    wrap.append(help);
  }
  return wrap;
}

function renderToggle(feature) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "switch";
  button.dataset.toggle = feature.name;
  button.setAttribute("role", "switch");
  const label = document.createElement("span");
  label.textContent = feature.label;
  const ui = document.createElement("span");
  ui.className = "switch-ui";
  ui.setAttribute("aria-hidden", "true");
  button.append(label, ui);
  syncToggle(button);
  return button;
}

function syncToggle(button) {
  const on = Number(state.values[button.dataset.toggle]) === 1;
  button.setAttribute("aria-checked", on ? "true" : "false");
  button.classList.toggle("is-on", on);
}

function renderRadio(feature) {
  const fieldset = document.createElement("fieldset");
  const legend = document.createElement("legend");
  legend.textContent = feature.label;
  fieldset.append(legend);
  for (const option of feature.options) {
    const label = document.createElement("label");
    label.className = "choice";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = feature.name;
    input.value = String(option.value);
    input.checked = Number(state.values[feature.name]) === option.value;
    label.append(input, document.createTextNode(option.label));
    fieldset.append(label);
  }
  return fieldset;
}

function renderSelect(feature) {
  const label = document.createElement("label");
  const title = document.createElement("span");
  title.className = "field-label";
  title.textContent = feature.label;
  const select = document.createElement("select");
  select.name = feature.name;
  for (const option of feature.options) {
    const node = document.createElement("option");
    node.value = String(option.value);
    node.textContent = option.label;
    node.selected = Number(state.values[feature.name]) === option.value;
    select.append(node);
  }
  label.append(title, select);
  return label;
}

function renderRange(feature) {
  const label = document.createElement("label");
  const header = document.createElement("span");
  header.className = "range-header";
  const title = document.createElement("span");
  title.className = "field-label";
  title.textContent = feature.label;
  const readout = document.createElement("span");
  readout.className = "readout";
  const input = document.createElement("input");
  const optionValues = feature.options.map((option) => option.value);
  input.type = feature.control === "number" ? "number" : "range";
  input.name = feature.name;
  input.min = String(feature.min_value ?? Math.min(...optionValues));
  input.max = String(feature.max_value ?? Math.max(...optionValues));
  input.step = "1";
  input.value = String(state.values[feature.name]);
  readout.textContent = readoutText(feature, input.value);
  header.append(title, readout);
  label.append(header, input);
  return label;
}

function readoutText(feature, value) {
  if (!feature.options.length) return String(value);
  const match = feature.options.find((option) => option.value === Number(value));
  return match ? match.label : String(value);
}

function onFormClick(event) {
  const button = event.target.closest("[data-toggle]");
  if (!button || !event.currentTarget.contains(button)) return;
  const name = button.dataset.toggle;
  state.values[name] = Number(state.values[name]) === 1 ? 0 : 1;
  syncToggle(button);
  onEdit();
}

function onFormInput(event) {
  const input = event.target;
  if (!input.name) return;
  const next = Number(input.value);
  if (Number.isNaN(next)) return;
  state.values[input.name] = next;
  const readout = input.closest(".field")?.querySelector(".readout");
  if (readout) {
    const feature = state.artifact.features.find((item) => item.name === input.name);
    readout.textContent = readoutText(feature, input.value);
  }
  onEdit();
}

function onEdit() {
  paint();
  if (state.editedSent) return;
  state.editedSent = true;
  track("answers_edited");
}

function featureByName(name) {
  return state.artifact.features.find((feature) => feature.name === name);
}

function describeValue(name, value) {
  const feature = featureByName(name);
  if (feature.options.length) {
    const match = feature.options.find((option) => option.value === Number(value));
    if (match) return match.label;
  }
  return String(Number(value));
}

function scoreLogistic(values) {
  const artifact = state.artifact;
  let logit = artifact.intercept;
  artifact.feature_columns.forEach((name, index) => {
    const scale = artifact.scale[index];
    const standardized = scale === 0 ? 0 : (Number(values[name]) - artifact.mean[index]) / scale;
    logit += artifact.coefficients[index] * standardized;
  });
  if (logit >= 0) return 1 / (1 + Math.exp(-logit));
  const expLogit = Math.exp(logit);
  return expLogit / (1 + expLogit);
}

function peerRate(sex, age) {
  const row = state.artifact.prevalence_by_group.find((item) => item.Sex === sex && item.Age === age);
  return row ? row.target : state.artifact.prevalence;
}

function drivers(values) {
  const base = scoreLogistic(values);
  return state.artifact.feature_columns
    .map((name) => {
      const contribution = base - scoreLogistic({ ...values, [name]: state.artifact.reference_profile[name] });
      return { name, contribution };
    })
    .filter((row) => Math.abs(row.contribution) >= 0.001)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 8);
}

function renderResultsShell() {
  const root = document.getElementById("results");
  root.replaceChildren();

  const metrics = document.createElement("div");
  metrics.className = "metrics";
  metrics.append(metricBlock("Estimated risk", "metric-risk"), metricBlock("Average for age and sex", "metric-peer", "metric-delta"));
  root.append(metrics);

  const callout = document.createElement("div");
  callout.id = "callout";
  root.append(callout);

  const comparisonCard = card("How this compares");
  const comparison = document.createElement("div");
  comparison.id = "comparison";
  comparisonCard.append(comparison);
  const comparisonNote = document.createElement("p");
  comparisonNote.className = "help";
  comparisonNote.textContent = "The mark on each bar is the screening threshold. The profile bar is the model’s probability. The other two are survey rates.";
  comparisonCard.append(comparisonNote);
  root.append(comparisonCard);

  const driverCard = card("What’s driving this estimate");
  const driverNote = document.createElement("p");
  driverNote.className = "help";
  driverNote.textContent =
    "Each bar shows how much the estimate changes if that one answer is replaced with the median answer in the training data. These are statistical associations, not causes.";
  const driverList = document.createElement("div");
  driverList.id = "drivers";
  driverCard.append(driverNote, driverList);
  root.append(driverCard);
}

function metricBlock(label, valueId, deltaId) {
  const block = document.createElement("div");
  block.className = "metric";
  const name = document.createElement("span");
  name.className = "metric-label";
  name.textContent = label;
  const value = document.createElement("strong");
  value.id = valueId;
  block.append(name, value);
  if (deltaId) {
    const delta = document.createElement("span");
    delta.id = deltaId;
    delta.className = "delta";
    block.append(delta);
  }
  return block;
}

function card(title) {
  const section = document.createElement("section");
  section.className = "card";
  const heading = document.createElement("h2");
  heading.textContent = title;
  section.append(heading);
  return section;
}

function paint() {
  const artifact = state.artifact;
  const values = state.values;
  const risk = scoreLogistic(values);
  const peer = peerRate(Number(values.Sex), Number(values.Age));
  const at = artifact.results["Logistic Regression"].test_at_threshold;
  const lift = at.precision / artifact.prevalence;

  document.getElementById("metric-risk").textContent = percent(risk);
  document.getElementById("metric-peer").textContent = percent(peer);
  const delta = document.getElementById("metric-delta");
  const points = (risk - peer) * 100;
  delta.textContent = `${points >= 0 ? "+" : ""}${points.toFixed(1)} pts vs peers`;
  delta.className = points > 0.05 ? "delta delta-up" : points < -0.05 ? "delta delta-down" : "delta";

  const callout = document.getElementById("callout");
  callout.className = risk >= artifact.threshold ? "callout callout-flag" : "callout callout-clear";
  callout.replaceChildren();
  const title = document.createElement("strong");
  if (risk >= artifact.threshold) {
    title.textContent = "Flagged for follow-up. ";
    callout.append(
      title,
      document.createTextNode(
        `The estimate is above the ${percent(artifact.threshold)} screening threshold. In testing, people above this line were ${lift.toFixed(1)}× more likely than average to have heart disease.`,
      ),
    );
  } else {
    title.textContent = "Below the screening threshold. ";
    callout.append(
      title,
      document.createTextNode(
        `The line is ${percent(artifact.threshold)}. In testing, ${percent(at.recall, 0)} of people with heart disease scored above it.`,
      ),
    );
  }

  paintComparison(risk, peer);
  paintDrivers(values);
  markPresets();

  const mobile = document.getElementById("mobile-risk");
  mobile.hidden = false;
  mobile.textContent = `Estimated risk ${percent(risk)}`;
}

function paintComparison(risk, peer) {
  const artifact = state.artifact;
  const domain = Math.max(0.25, risk, peer) * 1.15;
  const mark = `${(artifact.threshold / domain) * 100}%`;
  const rows = [
    ["This profile", risk, true],
    ["Same age group and sex", peer, false],
    ["All US adults surveyed", artifact.prevalence, false],
  ];
  const root = document.getElementById("comparison");
  root.replaceChildren();
  for (const [label, value, emphasis] of rows) {
    const row = document.createElement("div");
    row.className = "cmp-row";
    const name = document.createElement("span");
    name.textContent = label;
    const track = document.createElement("span");
    track.className = "cmp-track";
    track.style.setProperty("--mark", mark);
    const bar = document.createElement("span");
    bar.className = emphasis ? "cmp-bar is-profile" : "cmp-bar";
    bar.style.width = `${Math.min(100, (value / domain) * 100)}%`;
    track.append(bar);
    const number = document.createElement("span");
    number.className = "cmp-value";
    number.textContent = percent(value);
    row.append(name, track, number);
    root.append(row);
  }
}

function paintDrivers(values) {
  const root = document.getElementById("drivers");
  const rows = drivers(values);
  root.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement("p");
    empty.textContent = "This profile is close to the typical respondent on every factor.";
    root.append(empty);
    return;
  }
  const max = Math.max(...rows.map((row) => Math.abs(row.contribution)));
  for (const row of rows) {
    const feature = featureByName(row.name);
    const item = document.createElement("div");
    item.className = "driver";
    const label = document.createElement("span");
    label.textContent = `${feature.label}: ${describeValue(row.name, values[row.name])}`;
    const track = document.createElement("span");
    track.className = "driver-track";
    const bar = document.createElement("span");
    bar.className = row.contribution > 0 ? "driver-bar raises" : "driver-bar lowers";
    bar.style.width = `${(Math.abs(row.contribution) / max) * 100}%`;
    track.append(bar);
    const points = document.createElement("span");
    points.className = "driver-points";
    const pp = row.contribution * 100;
    points.textContent = `${pp >= 0 ? "+" : ""}${pp.toFixed(1)} pts`;
    item.append(label, track, points);
    root.append(item);
  }
}

function markPresets() {
  document.querySelectorAll("[data-preset]").forEach((button) => {
    const preset = state.artifact.presets[Number(button.dataset.preset)];
    const matches = Object.entries(preset.values).every(
      ([key, value]) => Number(state.values[key]) === Number(value),
    );
    button.classList.toggle("is-active", matches);
    button.setAttribute("aria-pressed", matches ? "true" : "false");
  });
}

function renderPerformance() {
  const artifact = state.artifact;
  const root = document.getElementById("performance");
  root.replaceChildren();
  const logistic = artifact.results["Logistic Regression"];
  const at = logistic.test_at_threshold;
  const matrix = at.confusion_matrix;
  const cases = matrix.tp + matrix.fn;
  const oneIn = Math.round(1 / at.precision);
  const lift = at.precision / artifact.prevalence;
  const boosting = artifact.results["Gradient Boosting"].test.roc_auc;
  const gap = Math.abs(boosting - logistic.test.roc_auc);

  const intro = document.createElement("p");
  intro.textContent = `All numbers are from a held-out test set of ${artifact.split_sizes.test.toLocaleString("en-US")} people that was not used for training, model selection, or choosing the threshold. This page scores ${artifact.scored_model}. ${artifact.selection_model} won validation PR-AUC and is ${gap.toFixed(3)} ROC-AUC ${boosting >= logistic.test.roc_auc ? "higher" : "lower"} on the test set, which is why the simpler model is the one that runs here.`;
  root.append(intro);

  const metrics = document.createElement("div");
  metrics.className = "metrics metrics-4";
  metrics.append(
    staticMetric("ROC-AUC", logistic.test.roc_auc.toFixed(3)),
    staticMetric("PR-AUC", logistic.test.pr_auc.toFixed(3)),
    staticMetric("Recall at threshold", percent(at.recall, 0)),
    staticMetric("Precision at threshold", percent(at.precision, 1)),
  );
  root.append(metrics);

  const plain = document.createElement("p");
  plain.textContent = `In plain terms, logistic regression flags ${percent(at.flagged_rate, 0)} of people and catches ${matrix.tp.toLocaleString("en-US")} of ${cases.toLocaleString("en-US")} people with heart disease. About 1 in ${oneIn} flagged people actually has it, which is ${lift.toFixed(1)}× the base rate. That trade-off fits a cheap first-pass screen followed by a real check-up. It is not a diagnosis.`;
  root.append(plain);

  root.append(resultsTable(artifact));
  root.append(coefficientCard(artifact));

  const figures = [
    ["roc_pr_curves.png", "ROC and precision–recall curves on the test set, for all three models."],
    ["calibration.png", "Calibration on the test set. A predicted risk should match the share of people who have heart disease."],
    ["threshold_tradeoff.png", "How the gradient boosting threshold was chosen on validation data. This page uses the logistic regression threshold instead."],
    ["confusion_matrix.png", "Test-set decisions for gradient boosting at its own threshold, not the model this page scores."],
    ["feature_importance.png", "Permutation importance for gradient boosting. Age and general health carry most of the signal."],
  ];
  const gallery = document.createElement("div");
  gallery.className = "gallery";
  for (const [file, caption] of figures) {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    image.src = `figures/${file}`;
    image.alt = caption;
    const cap = document.createElement("figcaption");
    cap.textContent = caption;
    figure.append(image, cap);
    gallery.append(figure);
  }
  root.append(gallery);
}

function staticMetric(label, value) {
  const block = document.createElement("div");
  block.className = "metric";
  const name = document.createElement("span");
  name.className = "metric-label";
  name.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  block.append(name, strong);
  return block;
}

function resultsTable(artifact) {
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  const head = document.createElement("tr");
  for (const label of ["Model", "ROC-AUC", "PR-AUC", "Brier score", "Recall", "Precision"]) {
    const cell = document.createElement("th");
    cell.textContent = label;
    head.append(cell);
  }
  table.append(head);
  for (const [name, result] of Object.entries(artifact.results)) {
    const row = document.createElement("tr");
    const label = document.createElement("td");
    label.textContent = modelLabel(artifact, name);
    row.append(label);
    for (const value of [
      result.test.roc_auc.toFixed(3),
      result.test.pr_auc.toFixed(3),
      result.test.brier.toFixed(4),
      percent(result.test_at_threshold.recall, 1),
      percent(result.test_at_threshold.precision, 1),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    table.append(row);
  }
  wrap.append(table);
  return wrap;
}

function modelLabel(artifact, name) {
  if (name === artifact.scored_model && name === artifact.selection_model) return `${name} (served)`;
  if (name === artifact.scored_model) return `${name} (this page)`;
  if (name === artifact.selection_model) return `${name} (selected)`;
  return name;
}

function coefficientCard(artifact) {
  const section = card("Standardized coefficients of the model this page runs");
  const note = document.createElement("p");
  note.className = "help";
  note.textContent =
    "Each answer is scaled to zero mean and unit variance before the coefficient is applied, so the lengths are comparable. A positive coefficient raises predicted risk.";
  const rows = artifact.feature_columns
    .map((name, index) => ({ name, coefficient: artifact.coefficients[index] }))
    .sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient))
    .slice(0, 8);
  const max = Math.max(...rows.map((row) => Math.abs(row.coefficient)));
  const list = document.createElement("div");
  list.className = "drivers";
  for (const row of rows) {
    const item = document.createElement("div");
    item.className = "driver";
    const label = document.createElement("span");
    label.textContent = featureByName(row.name).label;
    const track = document.createElement("span");
    track.className = "driver-track";
    const bar = document.createElement("span");
    bar.className = row.coefficient > 0 ? "driver-bar raises" : "driver-bar lowers";
    bar.style.width = `${(Math.abs(row.coefficient) / max) * 100}%`;
    track.append(bar);
    const value = document.createElement("span");
    value.className = "driver-points";
    value.textContent = `${row.coefficient >= 0 ? "+" : ""}${row.coefficient.toFixed(2)}`;
    item.append(label, track, value);
    list.append(item);
  }
  section.append(note, list);
  return section;
}

function renderAbout() {
  const artifact = state.artifact;
  const root = document.getElementById("about");
  const prevalence = percent(artifact.prevalence, 1);
  root.replaceChildren();
  root.append(
    paragraph(`Data. ${artifact.n_rows.toLocaleString("en-US")} responses to the CDC’s 2015 Behavioral Risk Factor Surveillance System (BRFSS) telephone survey of US adults. The target is whether the respondent reported ever having coronary heart disease or a heart attack (${prevalence} said yes).`),
    paragraph("Method. The data is split 60/20/20 into train, validation, and test. Logistic regression, random forest, and gradient boosting are compared. Gradient boosting hyperparameters are tuned with cross-validated random search on the training set. The model with the best validation PR-AUC is the selected model. The decision threshold for each model is the highest one that reaches 80% recall on validation data. No class weighting is used, so predicted risks stay calibrated."),
    paragraph(artifact.reason),
    list("Limitations.", [
      "Answers are self-reported, and the survey covers US adults in 2015 only.",
      "The target is having had heart disease, not developing it in the future. Some inputs, such as difficulty walking, may be consequences of heart disease rather than causes.",
      "The bars on the estimator are associations relative to a typical respondent, not causal effects.",
    ]),
    paragraph("Privacy. When analytics are configured, the page records anonymous usage events: a visit, which example profile was clicked, and whether any answer was edited. Answers and risk estimates are never sent. The database key used by the page can insert those events and cannot read them. With no analytics configuration, the estimate never leaves the browser."),
  );
  const source = document.createElement("p");
  const link = document.createElement("a");
  link.href = REPO_URL;
  link.textContent = REPO_URL.replace("https://", "");
  source.append(document.createTextNode("Source code, training pipeline, and tests: "), link);
  root.append(source);
}

function paragraph(text) {
  const node = document.createElement("p");
  node.textContent = text;
  return node;
}

function list(title, items) {
  const wrap = document.createElement("div");
  const heading = document.createElement("p");
  heading.textContent = title;
  const ul = document.createElement("ul");
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    ul.append(li);
  }
  wrap.append(heading, ul);
  return wrap;
}

function startAnalytics() {
  const params = new URLSearchParams(location.search);
  state.ref = cleanRef(params.get("ref"));
  const storageKey = "heart-risk-session";
  state.sessionId = sessionStorage.getItem(storageKey) || crypto.randomUUID();
  sessionStorage.setItem(storageKey, state.sessionId);
  track("session_start");
}

function cleanRef(value) {
  if (!value) return "direct";
  const ref = String(value).toLowerCase().replace(/[^a-z0-9_-]/g, "").slice(0, 40);
  return ref || "direct";
}

function track(event, properties = {}) {
  const config = window.ANALYTICS;
  if (!config || !config.url || !config.key || String(config.key).includes("YOUR_")) return;
  const headers = {
    apikey: config.key,
    "Content-Type": "application/json",
    Prefer: "return=minimal",
  };
  if (!String(config.key).startsWith("sb_")) headers.Authorization = `Bearer ${config.key}`;
  fetch(`${String(config.url).replace(/\/$/, "")}/rest/v1/events`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      session_id: state.sessionId,
      event,
      ref: state.ref,
      properties,
    }),
    keepalive: true,
  }).catch(() => {});
}

function percent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}
