/**
 * BILO Playground UI — TF Playground style: neuron heatmaps (t vs a), options, Pretrain/Finetune, loss components
 */

import * as d3 from "d3";
import {
  BILOModel,
  PINNModel,
  adamStepModel,
  adamStepA,
  type OdeType,
  type AdamState,
} from "./bilo";

const RECT_SIZE = 32;
const NETWORK_HEIGHT_MIN = 300;
/** Network fills the features column; width comes from container. */
const T_MIN = 0;
const T_MAX = 1;
const T_PLOT_MAX = 1; // u(t) only show t in [0, 1]
const N_PLOT = 101;
const HEATMAP_SAMPLES = 25;

let model: BILOModel | PINNModel;
let iter = 0;
let isPlaying = false;
let timerId: number | null = null;

// Options (from UI)
let lr = 0.001;
let lrA = 0.001;
let depthOption = 3; // 2 = one hidden layer, 3 = two hidden, …
let nHidden = 6;
let nPoints = 21;    // residual (collocation) points for L_res and L_grad
let nDataPoints = 11; // data points for L_data (finetune)
let wRes = 1;
let wGrad = 0.1;
let wData = 1.0;
let aParam = 1;      // network input a (pretrain fixed, finetune initial)
let aParamGT = 2;    // ground-truth a for generating training data (dashed line)
let noiseLevel = 0; // uniform noise added to training data: u += U(-noise, +noise)
let odeType: OdeType = "exponential";
let u0: number = 1;
let optimizer: "sgd" | "adam" = "adam";
let modelType: "bilo" | "pinn" = "bilo";
let mode: "pretrain" | "finetune" = "pretrain";
let a_learned = 1;   // only used in finetune
let adamState: AdamState = { m_W: [], v_W: [], m_b: [], v_b: [] };

let t_colloc: number[];
let a_colloc: number[];

// Training data for finetune (fixed until next reset; includes noise)
let trainingData: { t_data: number[]; u_data: number[] } | null = null;

/** Saved snapshot for load: weights, depth, n_hidden, and parameter a. */
let savedWeightsSnapshot: {
  depth: number;
  n_hidden: number;
  W: (number[][] | number[])[];
  b: (number[] | number)[];
  a: number;
} | null = null;


/** Residual/collocation points for L_res (and L_grad in BILO). */
function getResidualPoints(): number[] {
  return Array.from({ length: nPoints }, (_, i) => T_MIN + (T_MAX - T_MIN) * i / Math.max(1, nPoints - 1));
}

/** Data points for L_data (finetune); can differ from nPoints. */
function getDataPointsGrid(): number[] {
  return Array.from({ length: nDataPoints }, (_, i) => T_MIN + (T_MAX - T_MIN) * i / Math.max(1, nDataPoints - 1));
}

function buildModel() {
  if (modelType === "pinn") {
    model = new PINNModel(nHidden, depthOption, 42, odeType, u0);
  } else {
    model = new BILOModel(nHidden, depthOption, 42, odeType, u0);
  }
  t_colloc = getResidualPoints();
  a_learned = aParam;
  a_colloc = t_colloc.map(() => (mode === "pretrain" ? aParam : a_learned));
}

/** Analytical logistic solution: u(t) = u0*e^{at} / (1 + u0*(e^{at}-1)). */
function logisticSolution(t: number, a: number, u0Val: number): number {
  const eat = Math.exp(a * t);
  return (u0Val * eat) / (1 + u0Val * (eat - 1));
}

/** Training data for L_data only; uses nDataPoints. Respects odeType and u0. */
function generateTrainingData(): { t_data: number[]; u_data: number[] } {
  const t_data = getDataPointsGrid();
  const u_data = t_data.map(t => {
    const u =
      odeType === "exponential"
        ? u0 * Math.exp(aParamGT * t)
        : logisticSolution(t, aParamGT, u0);
    const noise = (2 * Math.random() - 1) * noiseLevel;
    return u + noise;
  });
  return { t_data, u_data };
}

function getDataForFinetune(): { t_data: number[]; u_data: number[]; a_data: number[] } | null {
  if (!trainingData) return null;
  const a_data = trainingData.t_data.map(() => a_learned);
  return { t_data: trainingData.t_data, u_data: trainingData.u_data, a_data };
}

function oneStep() {
  a_colloc = t_colloc.map(() => (mode === "pretrain" ? aParam : a_learned));

  let losses: { L_res: number; L_grad: number; L_data: number };
  let grads: Record<string, number[] | number[][] | number>;

  if (modelType === "pinn") {
    const pinnOpts: { w_res: number; w_data?: number; t_data?: number[]; u_data?: number[] } = { w_res: wRes };
    if (mode === "finetune") {
      const finetuneData = getDataForFinetune();
      if (finetuneData) {
        pinnOpts.w_data = wData;
        pinnOpts.t_data = finetuneData.t_data;
        pinnOpts.u_data = finetuneData.u_data;
      }
    }
    const out = (model as PINNModel).computeLossesAndGradientsPinn(t_colloc, a_colloc, pinnOpts);
    losses = out.losses;
    grads = out.grads;
  } else {
    const opts: { w_res: number; w_grad: number; w_data?: number; t_data?: number[]; u_data?: number[]; a_data?: number[] } = {
      w_res: wRes,
      w_grad: wGrad,
    };
    if (mode === "finetune") {
      const finetuneData = getDataForFinetune();
      if (finetuneData) {
        opts.w_data = wData;
        opts.t_data = finetuneData.t_data;
        opts.u_data = finetuneData.u_data;
        opts.a_data = finetuneData.a_data;
      }
    }
    const out = (model as BILOModel).computeLossesAndGradients(t_colloc, a_colloc, opts);
    losses = out.losses;
    grads = out.grads;
  }

  const step1 = iter + 1;
  if (optimizer === "adam") {
    adamStepModel(model as BILOModel, grads, adamState, step1, lr);
    if (mode === "finetune" && grads.a != null && typeof grads.a === "number") {
      a_learned = adamStepA(a_learned, grads.a, adamState, step1, lrA);
      a_learned = Math.max(0.1, Math.min(10, a_learned));
    }
  } else {
    const lrScaled = lr;
    for (let k = 0; k < model.depth; k++) {
      const Wk = model._W[k];
      const gWk = grads[`W${k + 1}`];
      if (Array.isArray(Wk[0])) {
        const W = Wk as number[][];
        const g = gWk as number[][];
        for (let i = 0; i < W.length; i++) for (let j = 0; j < W[i].length; j++) W[i][j] -= lrScaled * g[i][j];
      } else {
        const W = Wk as number[];
        const g = gWk as number[];
        for (let i = 0; i < W.length; i++) W[i] -= lrScaled * g[i];
      }
      const bk = model._b[k];
      const gbk = grads[`b${k + 1}`];
      if (typeof bk === "number") (model._b[k] as number) = bk - lrScaled * (gbk as number);
      else for (let i = 0; i < (bk as number[]).length; i++) (bk as number[])[i] -= lrScaled * (gbk as number[])[i];
    }
    if (mode === "finetune" && grads.a != null && typeof grads.a === "number") {
      a_learned = Math.max(0.1, Math.min(10, a_learned - lrA * grads.a));
    }
  }

  iter++;
  pushLossPoint(losses);
  updateUI();
}

type LossPoint = { step: number; L_res: number; L_grad: number; L_data: number; L_op?: number };
let lossHistory: LossPoint[] = [];
const LOSS_HISTORY_MAX = 500;

function pushLossPoint(losses: { L_res: number; L_grad: number; L_data: number }) {
  const L_op = wRes * losses.L_res + wGrad * losses.L_grad;
  lossHistory.push({
    step: iter,
    L_res: losses.L_res,
    L_grad: losses.L_grad,
    L_data: losses.L_data,
    L_op,
  });
  if (lossHistory.length > LOSS_HISTORY_MAX) lossHistory.shift();
}

const colorScale = d3.scale.linear<string, number>()
  .domain([-1, 0, 1])
  // .range(["#eae69e", "#83a561", "#1a4718"])
  .range(["#f59322", "#e8eaeb", "#0877bd"])
  .clamp(true);

/** Fill a canvas with a heatmap from 2D data (data[ix][iy]), color scale -1..1 */
function fillHeatmapCanvas(
  canvas: HTMLCanvasElement,
  data: number[][],
  size: number
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const ny = data.length;
  const nx = ny ? data[0].length : 0;
  if (nx === 0 || ny === 0) return;
  const image = ctx.createImageData(nx, ny);
  for (let iy = 0, p = -1; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const value = data[ix][iy];
      const c = d3.rgb(colorScale(value) as any);
      image.data[++p] = c.r;
      image.data[++p] = c.g;
      image.data[++p] = c.b;
      image.data[++p] = 220;
    }
  }
  ctx.putImageData(image, 0, 0);
}

/** Fill canvas with 2D data using 0..max linear scale mapped to orange-gray-blue */
function fillHeatmapCanvasU(canvas: HTMLCanvasElement, data: number[][]) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const ny = data.length;
  const nx = ny ? data[0].length : 0;
  if (nx === 0 || ny === 0) return;
  let minV = data[0][0];
  let maxV = data[0][0];
  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      const v = data[ix][iy];
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
  }
  const range = maxV - minV || 1;
  const image = ctx.createImageData(nx, ny);
  for (let iy = 0, p = -1; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const normalized = (data[ix][iy] - minV) / range;
      const value = 2 * normalized - 1;
      const c = d3.rgb(colorScale(value) as any);
      image.data[++p] = c.r;
      image.data[++p] = c.g;
      image.data[++p] = c.b;
      image.data[++p] = 220;
    }
  }
  ctx.putImageData(image, 0, 0);
}

function drawNetwork(container: d3.Selection<any>) {
  container.selectAll("*").remove();

  // Use container width (fills features column) like original TF Playground
  const parentEl = (container.node() as HTMLElement).parentElement;
  const networkWidth = parentEl ? Math.max(400, parentEl.clientWidth - 20) : 480;
  const networkHeight = Math.max(NETWORK_HEIGHT_MIN, parentEl ? parentEl.clientHeight - 120 : 0);

  const d = model.depth;
  const numHiddenLayers = d - 1;
  // Layers: [t, a] (or [t] for PINN) -> hidden_0 -> ... -> [N] -> [u]
  const layers: { ids: string[]; labels: string[]; layerIndex?: number }[] = [];
  layers.push(modelType === "pinn" ? { ids: ["t"], labels: ["t"] } : { ids: ["t", "a"], labels: ["t", "a"] });
  for (let L = 0; L < numHiddenLayers; L++) {
    const ids = Array.from({ length: nHidden }, (_, i) => `L${L}-${i}`);
    layers.push({ ids, labels: ids.map(() => ""), layerIndex: L }); // no numbers on heatmap
  }
  layers.push({ ids: ["N"], labels: ["N"] });
  layers.push({ ids: ["u"], labels: ["u"] });

  const padding = 10;
  const totalH = networkHeight - 2 * padding;
  const numLayers = layers.length;
  const layerWidth = (networkWidth - 2 * padding) / (numLayers + 1); // more layers => smaller spacing

  const node2coord: { [id: string]: { cx: number; cy: number } } = {};
  layers.forEach((layer, li) => {
    const nx = layer.ids.length;
    const spacing = nx > 1 ? Math.min(40, totalH / Math.max(1, nx - 1)) : 0;
    const startY = padding + (totalH - (nx - 1) * spacing) / 2;
    const cx = padding + (li + 1) * layerWidth;
    layer.ids.forEach((id, j) => {
      const cy = startY + j * spacing;
      node2coord[id] = { cx, cy };
    });
  });

  const tGrid = Array.from(
    { length: HEATMAP_SAMPLES },
    (_, i) => HEATMAP_T_DOMAIN[0] + (HEATMAP_T_DOMAIN[1] - HEATMAP_T_DOMAIN[0]) * i / (HEATMAP_SAMPLES - 1)
  );
  const currentA = mode === "pretrain" ? aParam : a_learned;
  const aMin = 0.5 * currentA;
  const aMax = 1.5 * currentA;
  const aGrid = Array.from(
    { length: HEATMAP_SAMPLES },
    (_, i) => aMin + (aMax - aMin) * i / (HEATMAP_SAMPLES - 1)
  );
  const sigmaGridAll = model.getSigmaGridAllLayers(tGrid, aGrid);
  const nGrid = model.getNGrid(tGrid, aGrid);
  const uGrid = model.getUGrid(tGrid, aGrid);

  // Input nodes t and a: mini heatmaps (built here, drawn after SVG so they sit on top)
  const tGrid2D = Array.from({ length: HEATMAP_SAMPLES }, (_, ix) =>
    Array.from({ length: HEATMAP_SAMPLES }, () => 2 * tGrid[ix] - 1)
  );
  const aRange = aMax - aMin;
  const aGrid2D = Array.from({ length: HEATMAP_SAMPLES }, (_, ix) =>
    Array.from({ length: HEATMAP_SAMPLES }, (_, iy) =>
      aRange === 0 ? 0 : ((aGrid[iy] - aMin) / aRange) * 2 - 1)
  );

  // Heatmaps for N and u
  ["N", "u"].forEach((id) => {
    const pos = node2coord[id];
    if (!pos) return;
    const left = pos.cx - RECT_SIZE / 2;
    const top = pos.cy - RECT_SIZE / 2;
    const div = container.insert("div", ":first-child")
      .attr("id", "canvas-" + id).attr("class", "canvas")
      .style("position", "absolute").style("left", left + "px").style("top", top + "px")
      .style("width", RECT_SIZE + "px").style("height", RECT_SIZE + "px");
    const canvas = div.append("canvas").attr("width", HEATMAP_SAMPLES).attr("height", HEATMAP_SAMPLES)
      .style("width", "100%").style("height", "100%");
    if (id === "N") fillHeatmapCanvas(canvas.node() as HTMLCanvasElement, nGrid, RECT_SIZE);
    else fillHeatmapCanvasU(canvas.node() as HTMLCanvasElement, uGrid);
  });

  // Heatmaps for every hidden neuron (each layer)
  for (let layerIdx = 0; layerIdx < numHiddenLayers; layerIdx++) {
    const grid = sigmaGridAll[layerIdx];
    if (!grid) continue;
    for (let j = 0; j < nHidden; j++) {
      const id = `L${layerIdx}-${j}`;
      const pos = node2coord[id];
      if (!pos) continue;
      const left = pos.cx - RECT_SIZE / 2;
      const top = pos.cy - RECT_SIZE / 2;
      const div = container.insert("div", ":first-child")
        .attr("id", "canvas-" + id).attr("class", "canvas")
        .style("position", "absolute").style("left", left + "px").style("top", top + "px")
        .style("width", RECT_SIZE + "px").style("height", RECT_SIZE + "px");
      const canvas = div.append("canvas").attr("width", HEATMAP_SAMPLES).attr("height", HEATMAP_SAMPLES)
        .style("width", "100%").style("height", "100%");
      fillHeatmapCanvas(canvas.node() as HTMLCanvasElement, grid[j], RECT_SIZE);
    }
  }

  const svg = container.append("svg").attr("width", networkWidth).attr("height", networkHeight);
  const g = svg.append("g").attr("class", "core");

  function cosLinkPath(x1: number, y1: number, x2: number, y2: number, n = 30): string {
    const dx = x2 - x1;
    const dy = y2 - y1;
  
    let d = `M ${x1},${y1}`;
    for (let i = 1; i <= n; i++) {
      const t = i / n;
      const x = x1 + dx * t;
      const s = (1 - Math.cos(Math.PI * t)) / 2; // flat slope at endpoints
      const y = y1 + dy * s;
      d += ` L ${x},${y}`;
    }
    return d;
  }

  function drawLink(fromId: string, toId: string, weight: number) {
  const from = node2coord[fromId];
  const to = node2coord[toId];
  if (!from || !to) return;

  const x1 = from.cx + RECT_SIZE / 2 + 2;
  const y1 = from.cy;
  const x2 = to.cx - RECT_SIZE / 2;
  const y2 = to.cy;

  const strokeW = Math.max(0.5, Math.abs(weight) * 3);

  g.insert("path", ":first-child")
    .attr("d", cosLinkPath(x1, y1, x2, y2))
    .attr("fill", "none")
    .attr("stroke", colorScale(Math.tanh(weight)))
    .attr("stroke-width", strokeW);
}

  // Links: input -> hidden0; hidden_k -> hidden_{k+1}; last hidden -> N; N -> u
  const W0 = model._W[0] as number[][];
  for (let j = 0; j < nHidden; j++) {
    drawLink("t", `L0-${j}`, W0[j][0]);
    if (modelType === "bilo") drawLink("a", `L0-${j}`, W0[j][1]);
  }
  for (let L = 0; L < numHiddenLayers - 1; L++) {
    const Wk = model._W[L + 1] as number[][];
    for (let i = 0; i < nHidden; i++) {
      for (let j = 0; j < nHidden; j++) {
        drawLink(`L${L}-${i}`, `L${L + 1}-${j}`, Wk[j][i]);
      }
    }
  }
  const Wlast = model._W[d - 1] as number[];
  for (let j = 0; j < nHidden; j++) {
    drawLink(`L${numHiddenLayers - 1}-${j}`, "N", Wlast[j]);
  }
  drawLink("N", "u", 1);

  // Node rects and labels (no numbers on heatmap nodes); t and a have heatmaps so rect fill none
  layers.forEach((layer, li) => {
    layer.ids.forEach((id, j) => {
      const { cx, cy } = node2coord[id];
      let fillColor = "#888";
      if (id === "t" || id === "a") fillColor = "none"; // heatmap shows through
      else if (layer.layerIndex !== undefined) {
        const lastHidden = layer.layerIndex === numHiddenLayers - 1;
        const w = lastHidden ? (model._W[d - 1] as number[])[j] : 0;
        fillColor = colorScale(Math.tanh(w)) as any;
      } else if (id === "N") fillColor = colorScale(0) as any;
      else if (id === "u") fillColor = "#333";
      g.append("rect")
        .attr("x", cx - RECT_SIZE / 2).attr("y", cy - RECT_SIZE / 2)
        .attr("width", RECT_SIZE).attr("height", RECT_SIZE)
        .attr("fill", fillColor).attr("stroke", "#333").attr("stroke-width", 1);
      g.append("text")
        .attr("x", cx).attr("y", cy + 4).attr("text-anchor", "middle")
        .attr("font-size", li === 0 ? "12px" : "10px")
        .text(layer.labels[j] || (id === "t" || id === "a" || id === "N" || id === "u" ? id : ""));
    });
  });

  // Input nodes t and a (a hidden for PINN): draw heatmap divs on top of SVG
  (modelType === "pinn" ? ["t"] : ["t", "a"]).forEach((id) => {
    const pos = node2coord[id];
    if (!pos) return;
    const left = pos.cx - RECT_SIZE / 2;
    const top = pos.cy - RECT_SIZE / 2;
    const grid = id === "t" ? tGrid2D : aGrid2D;
    const div = container.append("div")
      .attr("id", "canvas-" + id).attr("class", "canvas")
      .style("position", "absolute").style("left", left + "px").style("top", top + "px")
      .style("width", RECT_SIZE + "px").style("height", RECT_SIZE + "px").style("pointer-events", "none");
    const canvas = div.append("canvas").attr("width", HEATMAP_SAMPLES).attr("height", HEATMAP_SAMPLES)
      .style("width", "100%").style("height", "100%");
    fillHeatmapCanvas(canvas.node() as HTMLCanvasElement, grid, RECT_SIZE);
  });

  // Callouts: text below target, arrow pointing up (marker so head follows path)
  const calloutWidth = 130;
  let calloutMarkerCounter = 0;
  function addCalloutBelow(nodeCx: number, nodeCy: number, labelText: string) {
    const topPx = nodeCy + RECT_SIZE / 2 + 6;
    const leftPx = Math.max(4, nodeCx - calloutWidth / 2);
    const div = container.append("div").attr("class", "bilo-callout bilo-callout-below")
      .style("position", "absolute").style("left", leftPx + "px").style("top", topPx + "px")
      .style("width", calloutWidth + "px").style("pointer-events", "none");
    const svg = div.append("svg").attr("class", "bilo-callout-arrow-up").attr("viewBox", "0 0 24 20").attr("width", 24).attr("height", 20);
    const markerId = `bilo-callout-marker-${calloutMarkerCounter++}`;
    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", markerId)
      .attr("markerWidth", 4)
      .attr("markerHeight", 4)
      .attr("refX", 2)
      .attr("refY", 2)
      .attr("orient", "auto")
      .attr("markerUnits", "userSpaceOnUse")
      .append("path")
      .attr("d", "M0,0 L4,2 L0,4 z")
      .attr("fill", "#333")
      .attr("fill-opacity", "0.5");
    svg.append("path")
      .attr("d", "M12 18 Q10 10 12 2")
      .attr("fill", "none")
      .attr("stroke", "#333")
      .attr("stroke-opacity", "0.5")
      .attr("stroke-width", "1.5")
      .attr("marker-end", `url(#${markerId})`);
    div.append("div").attr("class", "bilo-callout-label").text(labelText);
  }

  const pad = 6;
  // PINN: t is only input — below t node
  if (modelType === "pinn") {
    const tPos = node2coord["t"];
    if (tPos) addCalloutBelow(tPos.cx, tPos.cy, "In PINN, t is the only input to the network.");
  }
  // BILO: a is also input — below a node
  if (modelType === "bilo") {
    const aPos = node2coord["a"];
    if (aPos) addCalloutBelow(aPos.cx, aPos.cy, "In BiLO, a is also input, but only evaluated at current a.");
  }
  // Output u: u = tN + u₀ — below u node
  const uPos = node2coord["u"];
  if (uPos) addCalloutBelow(uPos.cx, uPos.cy, "u = tN + u₀");
}

function resetLossChart() {
  lossHistory = [];
}

function redrawLossChart() {
  const container = d3.select("#loss-chart");
  container.selectAll("*").remove();
  if (lossHistory.length < 1) return;

  const node = container.node() as HTMLElement;
  const w = node.offsetWidth || 400;
  const h = node.offsetHeight || 140;
  const margin = { top: 8, right: 72, bottom: 22, left: 48 };
  const width = w - margin.left - margin.right;
  const height = h - margin.top - margin.bottom;

  const xScale = d3.scale.linear()
    .domain([Math.max(0, iter - LOSS_HISTORY_MAX), iter])
    .range([0, width]);

  // Build series based on model and mode
  const isPretrain = mode === "pretrain";
  const series: { key: string; color: string; getVal: (d: LossPoint) => number }[] = [];
  if (modelType === "bilo") {
    series.push({ key: "L_res", color: "#f59322", getVal: d => d.L_res });
    series.push({ key: "L_grad", color: "#0877bd", getVal: d => d.L_grad });
    series.push({ key: "L_op", color: "#333", getVal: d => (d.L_op ?? wRes * d.L_res + wGrad * d.L_grad) });
    if (!isPretrain) series.push({ key: "L_data", color: "#0a0", getVal: d => d.L_data });
  } else {
    series.push({ key: "L_res", color: "#f59322", getVal: d => d.L_res });
    if (!isPretrain) series.push({ key: "L_data", color: "#0a0", getVal: d => d.L_data });
  }

  const allVals: number[] = [];
  lossHistory.forEach(d => series.forEach(s => allVals.push(s.getVal(d))));
  const minV = Math.max(1e-8, d3.min(allVals)!);
  const maxV = d3.max(allVals)!;
  const yScale = d3.scale.log()
    .domain([minV, maxV])
    .range([height, 0]);

  const svg = container.append("svg").attr("width", w).attr("height", h)
    .append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  series.forEach((s) => {
    const line = d3.svg.line<LossPoint>()
      .x(d => xScale(d.step))
      .y(d => yScale(Math.max(minV, s.getVal(d))));
    svg.append("path")
      .datum(lossHistory)
      .attr("fill", "none")
      .attr("stroke", s.color)
      .attr("stroke-width", s.key === "L_op" ? 1.5 : 1)
      .attr("stroke-dasharray", s.key === "L_op" ? "0" : "2,2")
      .attr("d", line);
  });

  const xAxis = d3.svg.axis().scale(xScale).orient("bottom").ticks(5).tickSize(2);
  const yAxis = d3.svg.axis()
    .scale(yScale)
    .orient("left")
    .tickValues(d3.range(3).map(i => {
      const logMin = Math.log10(minV);
      const logMax = Math.log10(maxV);
      return Math.pow(10, logMin + (logMax - logMin) * i / 2);
    }))
    .tickFormat(d3.format(".0e"))
    .tickSize(2);
  svg.append("g").attr("class", "axis").attr("transform", "translate(0," + height + ")").call(xAxis);
  svg.append("g").attr("class", "axis y-axis").call(yAxis);

  const legendLineH = 16;
  const legend = svg.append("g").attr("transform", "translate(" + (width + 2) + ",0)");
  const subLabels: Record<string, string> = { L_res: "res", L_grad: "grad", L_data: "data", L_op: "op" };
  series.forEach((s, i) => {
    legend.append("line").attr("x1", 0).attr("y1", i * legendLineH).attr("x2", 10).attr("y2", i * legendLineH)
      .attr("stroke", s.color).attr("stroke-width", s.key === "L_op" ? 1.5 : 1);
    const txt = legend.append("text").attr("x", 14).attr("y", i * legendLineH + 4).attr("font-size", "12px");
    const sub = subLabels[s.key] || s.key;
    txt.append("tspan").text("L");
    txt.append("tspan").attr("baseline-shift", "sub").attr("font-size", "10px").text(sub);
  });
}

const HEATMAP_T_DOMAIN: [number, number] = [0, T_PLOT_MAX];

let plotSvg: d3.Selection<any> | null = null;

function redrawPlot() {
  const container = d3.select("#plot-1d");
  container.selectAll("*").remove();

  const node = container.node() as HTMLElement;
  const w = node.offsetWidth || 500;
  const h = node.offsetHeight || 240;
  const margin = { top: 8, right: 120, bottom: 28, left: 40 };
  const width = w - margin.left - margin.right;
  const height = h - margin.top - margin.bottom;

  const aDisplay = mode === "pretrain" ? aParam : a_learned;
  const tPlot = Array.from({ length: N_PLOT }, (_, i) => 0 + (T_PLOT_MAX - 0) * i / (N_PLOT - 1));
  const uPred = model.evalUArray(tPlot, aDisplay);  // PINNModel.evalU ignores second arg
  const uExactGT = mode === "finetune" ? tPlot.map(t =>
    odeType === "exponential" ? u0 * Math.exp(aParamGT * t) : logisticSolution(t, aParamGT, u0)
  ) : null;
  const uAnalyticalCurrent = tPlot.map(t =>
    odeType === "exponential" ? u0 * Math.exp(aDisplay * t) : logisticSolution(t, aDisplay, u0)
  );

  let yMax = Math.max(d3.max(uPred)!, d3.max(uAnalyticalCurrent)!);
  if (uExactGT) yMax = Math.max(yMax, d3.max(uExactGT)!);
  if (trainingData && trainingData.u_data.length > 0) {
    const dataMax = d3.max(trainingData.u_data)!;
    if (dataMax > yMax) yMax = dataMax;
  }
  const xScale = d3.scale.linear().domain([0, T_PLOT_MAX]).range([0, width]);
  const yScale = d3.scale.linear()
    .domain([0, yMax * 1.05])
    .range([height, 0]);

  plotSvg = container.append("svg").attr("width", w).attr("height", h)
    .append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  const line = d3.svg.line<number>().x((_, i) => xScale(tPlot[i])).y(d => yScale(d));

  plotSvg.append("path")
    .datum(uPred)
    .attr("fill", "none")
    .attr("stroke", "#0877bd")
    .attr("stroke-width", 2)
    .attr("d", line);

  plotSvg.append("path")
    .datum(uAnalyticalCurrent)
    .attr("fill", "none")
    .attr("stroke", "#666")
    .attr("stroke-width", 1.5)
    .attr("d", line);

  if (uExactGT) {
    plotSvg.append("path")
      .datum(uExactGT)
      .attr("fill", "none")
      .attr("stroke", "#333")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "4,2")
      .attr("d", line);
  }

  if (trainingData && trainingData.t_data.length > 0) {
    for (let i = 0; i < trainingData.t_data.length; i++) {
      plotSvg.append("circle")
        .attr("class", "train-data")
        .attr("cx", xScale(trainingData.t_data[i]))
        .attr("cy", yScale(trainingData.u_data[i]))
        .attr("r", 4)
        .attr("fill", "#c00")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1);
    }
  }

  const xAxis = d3.svg.axis().scale(xScale).orient("bottom").ticks(6).tickSize(2);
  const yAxis = d3.svg.axis().scale(yScale).orient("left").ticks(5).tickSize(2);
  plotSvg.append("g").attr("class", "axis").attr("transform", "translate(0," + height + ")").call(xAxis);
  plotSvg.append("g").attr("class", "axis").call(yAxis);

  const legendY = 14;
  const modelLabel = modelType === "pinn" ? "PINN" : "BILO";
  const aStr = aDisplay.toFixed(2);
  plotSvg.append("line").attr("x1", width + 8).attr("y1", legendY).attr("x2", width + 28).attr("y2", legendY)
    .attr("stroke", "#0877bd").attr("stroke-width", 2);
  plotSvg.append("text").attr("x", width + 32).attr("y", legendY + 4).attr("font-size", "11px")
    .text(modelLabel + " (a=" + aStr + ")");
  plotSvg.append("line").attr("x1", width + 8).attr("y1", legendY + 18).attr("x2", width + 28).attr("y2", legendY + 18)
    .attr("stroke", "#666").attr("stroke-width", 1.5);
  plotSvg.append("text").attr("x", width + 32).attr("y", legendY + 22).attr("font-size", "11px")
    .text("Exact (a=" + aStr + ")");
  if (uExactGT) {
    plotSvg.append("line").attr("x1", width + 8).attr("y1", legendY + 36).attr("x2", width + 28).attr("y2", legendY + 36)
      .attr("stroke", "#333").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,2");
    plotSvg.append("text").attr("x", width + 32).attr("y", legendY + 40).attr("font-size", "11px")
      .text("GT (a=" + aParamGT + ")");
  }
  if (trainingData && trainingData.t_data.length > 0) {
    plotSvg.append("circle").attr("cx", width + 18).attr("cy", legendY + 54).attr("r", 3)
      .attr("fill", "#c00").attr("stroke", "#fff").attr("stroke-width", 1);
    plotSvg.append("text").attr("x", width + 32).attr("y", legendY + 58).attr("font-size", "11px").text("Data");
  }
}

function applyPretrainRestrictions() {
  const isPretrain = mode === "pretrain";
  const aParamLabel = document.querySelector("label[for='aParam']");
  if (aParamLabel) aParamLabel.textContent = isPretrain ? "Fixed a" : "Learned a";
  const calloutModeText = document.getElementById("callout-mode-text");
  if (calloutModeText) {
    calloutModeText.textContent = isPretrain
      ? "Pretrain: a is fixed, residual only."
      : "Finetune: a is learned; fit data.";
  }

  const aParamGTInput = d3.select("#aParamGT").node() as HTMLInputElement;
  const nDataPointsInput = d3.select("#nDataPoints").node() as HTMLInputElement;
  const wGradControl = d3.select("#wGradControl");

  const aParamGTParent = aParamGTInput?.parentElement;
  const nDataPointsParent = nDataPointsInput?.parentElement;

  if (isPretrain) {
    if (aParamGTInput) {
      aParamGTInput.disabled = true;
      aParamGTInput.value = "";
      aParamGTInput.placeholder = "—";
    }
    aParamGTParent?.classList.add("bilo-pretrain-disabled");
    if (nDataPointsInput) {
      nDataPointsInput.disabled = true;
      nDataPointsInput.value = "";
      nDataPointsInput.placeholder = "—";
    }
    nDataPointsParent?.classList.add("bilo-pretrain-disabled");
    wGradControl.classed("bilo-pretrain-disabled", true);
    const wGradInput = wGradControl.select("input").node() as HTMLInputElement;
    if (wGradInput) {
      wGradInput.disabled = true;
      wGradInput.value = "";
      wGradInput.placeholder = "—";
    }
  } else {
    if (aParamGTInput) {
      aParamGTInput.disabled = false;
      aParamGTInput.value = String(aParamGT);
      aParamGTInput.placeholder = "";
    }
    aParamGTParent?.classList.remove("bilo-pretrain-disabled");
    if (nDataPointsInput) {
      nDataPointsInput.disabled = false;
      nDataPointsInput.value = String(nDataPoints);
      nDataPointsInput.placeholder = "";
    }
    nDataPointsParent?.classList.remove("bilo-pretrain-disabled");
    wGradControl.classed("bilo-pretrain-disabled", false);
    const wGradInput = wGradControl.select("input").node() as HTMLInputElement;
    if (wGradInput) {
      wGradInput.disabled = false;
      wGradInput.value = String(wGrad);
      wGradInput.placeholder = "";
    }
  }
}

function updateUI() {
  ensureTrainingData();
  d3.select("#iter-number").text(iter);
  applyPretrainRestrictions();
  // Sync parameter a control with network: in finetune show a_learned so user sees updates
  const aInput = d3.select("#aParam").node() as HTMLInputElement;
  if (aInput) {
    const aVal = mode === "finetune" ? a_learned : aParam;
    aInput.value = String(Math.round(aVal * 1000) / 1000);
  }
  drawNetwork(d3.select("#network"));
  redrawPlot();
  redrawLossChart();
}

function reset() {
  iter = 0;
  a_learned = aParam;
  adamState = { m_W: [], v_W: [], m_b: [], v_b: [] };
  buildModel();
  trainingData = mode === "finetune" ? generateTrainingData() : null;
  resetLossChart();
  updateUI();
}

function ensureTrainingData() {
  if (mode === "finetune" && !trainingData) {
    trainingData = generateTrainingData();
  }
}

function play() {
  if (isPlaying) return;
  isPlaying = true;
  d3.select("#play-pause-button").classed("playing", true);
  d3.select("#play-pause-button i").style("display", (_, i) => (i === 0 ? "none" : "inline"));
  const tick = () => {
    if (!isPlaying) return;
    oneStep();
    timerId = window.setTimeout(tick, 16) as unknown as number;
  };
  tick();
}

function pause() {
  isPlaying = false;
  if (timerId != null) window.clearTimeout(timerId);
  timerId = null;
  d3.select("#play-pause-button").classed("playing", false);
  d3.select("#play-pause-button i").style("display", (_, i) => (i === 0 ? "inline" : "none"));
}

function getToggleValue(toggleId: string): string {
  const btn = document.querySelector(`#${toggleId} .bilo-toggle-btn.active`);
  return (btn && (btn as HTMLElement).getAttribute("data-value")) || "";
}

function setToggleActive(toggleId: string, value: string) {
  d3.selectAll(`#${toggleId} .bilo-toggle-btn`).classed("active", false);
  d3.select(`#${toggleId} .bilo-toggle-btn[data-value="${value}"]`).classed("active", true);
}

function syncOptionsFromUI() {
  odeType = getToggleValue("odeToggle") as OdeType || "exponential";
  u0 = Math.max(0.01, Math.min(2, +(d3.select("#u0").node() as HTMLInputElement).value || 1));
  optimizer = (getToggleValue("optimizerToggle") as "sgd" | "adam") || "adam";
  modelType = (getToggleValue("modelTypeToggle") as "bilo" | "pinn") || "bilo";
  lr = +(d3.select("#learningRate").node() as HTMLInputElement).value || 0.001;
  lrA = +(d3.select("#lrA").node() as HTMLInputElement).value || 0.001;
  // depthOption and nHidden are updated by +/- buttons; sync from display if present
  const depthEl = document.getElementById("depth-value");
  if (depthEl) {
    const numLayers = parseInt(depthEl.textContent || "1", 10);
    depthOption = Math.max(2, Math.min(4, numLayers + 1)); // numLayers 1..3 => depth 2..4
  }
  const nHiddenEl = document.getElementById("nHidden-value");
  if (nHiddenEl) {
    nHidden = Math.max(2, Math.min(8, parseInt(nHiddenEl.textContent || "6", 10)));
  }
  nPoints = Math.max(3, Math.min(101, +(d3.select("#nPoints").node() as HTMLInputElement).value || 21));
  mode = (getToggleValue("modeToggle") as "pretrain" | "finetune") || "pretrain";
  if (mode === "finetune") {
    nDataPoints = Math.max(2, Math.min(101, +(d3.select("#nDataPoints").node() as HTMLInputElement).value || 11));
  }
  const wResVal = parseFloat((d3.select("#wRes").node() as HTMLInputElement).value);
  wRes = !isNaN(wResVal) && wResVal >= 0 ? wResVal : 1;
  if (mode === "finetune" && modelType === "bilo") {
    const wGradVal = parseFloat((d3.select("#wGrad").node() as HTMLInputElement).value);
    wGrad = !isNaN(wGradVal) && wGradVal >= 0 ? wGradVal : 0.1;
  }
  const wDataVal = parseFloat((d3.select("#wData").node() as HTMLInputElement).value);
  wData = !isNaN(wDataVal) && wDataVal >= 0 ? wDataVal : 1.0;
  aParam = Math.max(0.1, Math.min(10, +(d3.select("#aParam").node() as HTMLInputElement).value || 1));
  if (mode === "finetune") a_learned = aParam; // sync from control when in finetune
  if (mode === "finetune") aParamGT = +(d3.select("#aParamGT").node() as HTMLInputElement).value || 2;
  noiseLevel = Math.max(0, Math.min(0.5, +(d3.select("#noise").node() as HTMLInputElement).value || 0));
  const wGradControl = d3.select("#wGradControl");
  if (modelType === "pinn") wGradControl.style("opacity", "0.5").style("pointer-events", "none");
  else wGradControl.style("opacity", "1").style("pointer-events", "auto");
}

function init() {
  syncOptionsFromUI();
  d3.select("#depth-value").text(String(depthOption - 1));
  d3.select("#nHidden-value").text(String(nHidden));
  buildModel();
  updateUI();

  d3.select("#reset-button").on("click", () => { pause(); reset(); });
  d3.select("#save-weights-btn").on("click", () => {
    const ver = model.exportForVerification();
    savedWeightsSnapshot = {
      depth: ver.depth,
      n_hidden: ver.n_hidden,
      W: ver.W,
      b: ver.b,
      a: mode === "finetune" ? a_learned : aParam,
    };
  });
  d3.select("#load-weights-btn").on("click", () => {
    if (!savedWeightsSnapshot) return;
    if (savedWeightsSnapshot.depth !== model.depth || savedWeightsSnapshot.n_hidden !== model.n_hidden) return;
    const ok = model.setWeights(savedWeightsSnapshot);
    if (!ok) return;
    aParam = savedWeightsSnapshot.a;
    a_learned = savedWeightsSnapshot.a;
    const aInput = d3.select("#aParam").node() as HTMLInputElement;
    if (aInput) aInput.value = String(aParam);
    a_colloc = t_colloc.map(() => (mode === "pretrain" ? aParam : a_learned));
    redrawPlot();
    redrawLossChart();
  });
  d3.select("#play-pause-button").on("click", function() {
    if (isPlaying) pause(); else play();
    d3.select(this).classed("playing", isPlaying);
  });
  d3.select("#step-button").on("click", () => { pause(); oneStep(); });

  d3.select("#learningRate").on("change", function() { syncOptionsFromUI(); });
  d3.select("#lrA").on("change", function() { syncOptionsFromUI(); });

  function onToggleClick(toggleId: string, handler: (value: string) => void) {
    d3.selectAll(`#${toggleId} .bilo-toggle-btn`).on("click", function() {
      const val = (this as HTMLElement).getAttribute("data-value") || "";
      setToggleActive(toggleId, val);
      handler(val);
    });
  }

  onToggleClick("odeToggle", (val) => {
    odeType = val as OdeType;
    const u0Input = document.getElementById("u0") as HTMLInputElement;
    if (u0Input) u0Input.value = odeType === "logistic" ? "0.1" : "1";
    syncOptionsFromUI();
    pause();
    reset();
  });

  onToggleClick("modelTypeToggle", () => {
    syncOptionsFromUI();
    pause();
    reset();
  });

  onToggleClick("optimizerToggle", () => {
    syncOptionsFromUI();
    adamState = { m_W: [], v_W: [], m_b: [], v_b: [] };
  });

  onToggleClick("modeToggle", () => {
    syncOptionsFromUI();
    mode = getToggleValue("modeToggle") as "pretrain" | "finetune";
    a_learned = aParam;
    a_colloc = t_colloc.map(() => (mode === "pretrain" ? aParam : a_learned));
    if (mode === "finetune") trainingData = generateTrainingData();
    else trainingData = null;
    applyPretrainRestrictions();
    redrawPlot();
  });
  d3.select("#u0").on("change", function() {
    syncOptionsFromUI();
    pause();
    reset();
  });

  // +/- buttons for hidden layers (depth 2 = 1 layer, 3 = 2, 4 = 3)
  function updateDepthDisplay() {
    d3.select("#depth-value").text(String(depthOption - 1));
  }
  d3.select("#depth-plus").on("click", () => {
    if (depthOption < 4) { depthOption++; updateDepthDisplay(); pause(); reset(); }
  });
  d3.select("#depth-minus").on("click", () => {
    if (depthOption > 2) { depthOption--; updateDepthDisplay(); pause(); reset(); }
  });

  // +/- buttons for neurons per layer
  function updateNHiddenDisplay() {
    d3.select("#nHidden-value").text(String(nHidden));
  }
  d3.select("#nHidden-plus").on("click", () => {
    if (nHidden < 8) { nHidden++; updateNHiddenDisplay(); pause(); reset(); }
  });
  d3.select("#nHidden-minus").on("click", () => {
    if (nHidden > 2) { nHidden--; updateNHiddenDisplay(); pause(); reset(); }
  });
  d3.select("#nPoints").on("change", function() {
    syncOptionsFromUI();
    t_colloc = getResidualPoints();
    a_colloc = t_colloc.map(() => (mode === "pretrain" ? aParam : a_learned));
    if (mode === "finetune") trainingData = generateTrainingData();
  });
  d3.select("#nDataPoints").on("change", function() {
    syncOptionsFromUI();
    if (mode === "finetune") trainingData = generateTrainingData();
    redrawPlot();
  });
  d3.select("#aParamGT").on("change", function() {
    syncOptionsFromUI();
    if (mode === "finetune") trainingData = generateTrainingData();
    redrawPlot();
  });
  d3.select("#noise").on("change", function() {
    syncOptionsFromUI();
    if (mode === "finetune") trainingData = generateTrainingData();
    redrawPlot();
  });
  d3.select("#wRes").on("change", () => syncOptionsFromUI());
  d3.select("#wGrad").on("change", () => syncOptionsFromUI());
  d3.select("#wData").on("change", () => syncOptionsFromUI());
  d3.select("#aParam").on("change", function() {
    const v = +(this as HTMLInputElement).value || 1;
    aParam = Math.max(0.1, Math.min(10, v));
    if (mode === "finetune") a_learned = aParam;
    a_colloc = t_colloc.map(() => (mode === "pretrain" ? aParam : a_learned));
    redrawPlot();
  });

  window.addEventListener("resize", () => { redrawPlot(); redrawLossChart(); });
}

init();
