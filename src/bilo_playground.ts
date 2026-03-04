/**
 * BILO Playground UI — TF Playground style: neuron heatmaps (t vs a), options, Pretrain/Finetune, loss components
 */

import * as d3 from "d3";
import { BILOModel, runBiloTests } from "./bilo";

const RECT_SIZE = 32;
const NETWORK_HEIGHT = 320;
/** Width grows with depth so layers fit; min for 2 layers. */
function getNetworkWidth(): number {
  const d = model ? model.depth : 2;
  return Math.max(380, 80 * (d + 2));
}
const T_MIN = 0;
const T_MAX = 1;
const T_PLOT_MAX = 1; // u(t) only show t in [0, 1]
const N_PLOT = 101;
const HEATMAP_SAMPLES = 25;

let model: BILOModel;
let iter = 0;
let isPlaying = false;
let timerId: number | null = null;

// Options (from UI)
let lr = 0.02;
let lrA = 0.001;
let depthOption = 2; // 2 = one hidden layer, 3 = two hidden, …
let nHidden = 4;
let nPoints = 21;    // same points for residual and data loss
let wRes = 1;
let wGrad = 0.1;
let wData = 0.5;
let aParam = 1;      // network input a (pretrain fixed, finetune initial)
let aParamGT = 2;    // ground-truth a for generating training data (dashed line)
let noiseLevel = 0; // uniform noise added to training data: u += U(-noise, +noise)
let mode: "pretrain" | "finetune" = "pretrain";
let a_learned = 1;   // only used in finetune

let t_colloc: number[];
let a_colloc: number[];

// Training data for finetune (fixed until next reset; includes noise)
let trainingData: { t_data: number[]; u_data: number[] } | null = null;

// Run tests on load
{
  const { pass, messages } = runBiloTests();
  d3.select("#test-status").html(
    messages.join("<br>") + (pass ? " <span style='color:green'>All tests passed.</span>" : " <span style='color:red'>Some tests failed.</span>")
  );
}

function getTPoints(): number[] {
  return Array.from({ length: nPoints }, (_, i) => T_MIN + (T_MAX - T_MIN) * i / (nPoints - 1));
}

function buildModel() {
  model = new BILOModel(nHidden, depthOption, 42);
  t_colloc = getTPoints();
  a_learned = aParam;
  a_colloc = t_colloc.map(() => (mode === "pretrain" ? aParam : a_learned));
}

/** Training data on the same grid as t_colloc (for residual + data loss). */
function generateTrainingData(): { t_data: number[]; u_data: number[] } {
  const t_data = t_colloc.slice();
  const u_data = t_data.map(t => {
    const u = Math.exp(aParamGT * t);
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

  const { losses, grads } = model.computeLossesAndGradients(t_colloc, a_colloc, opts);

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
    a_learned = Math.max(0.1, Math.min(3, a_learned - lrA * grads.a));
  }

  iter++;
  pushLossPoint(losses);
  updateUI();
}

type LossPoint = { step: number; L_res: number; L_grad: number; L_data: number };
let lossHistory: LossPoint[] = [];
const LOSS_HISTORY_MAX = 500;

function pushLossPoint(losses: { L_res: number; L_grad: number; L_data: number }) {
  lossHistory.push({
    step: iter,
    L_res: losses.L_res,
    L_grad: losses.L_grad,
    L_data: losses.L_data,
  });
  if (lossHistory.length > LOSS_HISTORY_MAX) lossHistory.shift();
}

const colorScale = d3.scale.linear<string, number>()
  .domain([-1, 0, 1])
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

  const d = model.depth;
  const numHiddenLayers = d - 1;
  // Layers: [t, a] -> hidden_0 -> ... -> hidden_{d-2} -> [N] -> [u]
  const layers: { ids: string[]; labels: string[]; layerIndex?: number }[] = [];
  layers.push({ ids: ["t", "a"], labels: ["t", "a"] });
  for (let L = 0; L < numHiddenLayers; L++) {
    const ids = Array.from({ length: nHidden }, (_, i) => `L${L}-${i}`);
    const Wnext = model._W[L + 1];
    const isLastHidden = L === numHiddenLayers - 1;
    const labels = ids.map((_, j) => {
      if (isLastHidden) return ((Wnext as number[])[j] * 100).toFixed(0);
      const row = (Wnext as number[][])[j];
      return row ? (row[0] * 100).toFixed(0) : "";
    });
    layers.push({ ids, labels, layerIndex: L });
  }
  layers.push({ ids: ["N"], labels: ["N"] });
  layers.push({ ids: ["u"], labels: ["u"] });

  const networkWidth = getNetworkWidth();
  const padding = 24;
  const totalH = NETWORK_HEIGHT - 2 * padding;
  const numLayers = layers.length;
  const layerWidth = Math.min(56, (networkWidth - 2 * padding) / (numLayers + 1));

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
  const aGrid = Array.from(
    { length: HEATMAP_SAMPLES },
    (_, i) => HEATMAP_A_DOMAIN[0] + (HEATMAP_A_DOMAIN[1] - HEATMAP_A_DOMAIN[0]) * i / (HEATMAP_SAMPLES - 1)
  );
  const sigmaGridAll = model.getSigmaGridAllLayers(tGrid, aGrid);
  const nGrid = model.getNGrid(tGrid, aGrid);
  const uGrid = model.getUGrid(tGrid, aGrid);

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

  const svg = container.append("svg").attr("width", networkWidth).attr("height", NETWORK_HEIGHT);
  const g = svg.append("g").attr("class", "core");

  function drawLink(fromId: string, toId: string, weight: number) {
    const from = node2coord[fromId];
    const to = node2coord[toId];
    if (!from || !to) return;
    const x1 = from.cx + RECT_SIZE / 2;
    const y1 = from.cy;
    const x2 = to.cx - RECT_SIZE / 2;
    const y2 = to.cy;
    const strokeW = Math.max(0.5, Math.abs(weight) * 3);
    g.insert("line", ":first-child")
      .attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
      .attr("stroke", colorScale(Math.tanh(weight)) as any).attr("stroke-width", strokeW);
  }

  // Links: input -> hidden0; hidden_k -> hidden_{k+1}; last hidden -> N; N -> u
  const W0 = model._W[0] as number[][];
  for (let j = 0; j < nHidden; j++) {
    drawLink("t", `L0-${j}`, W0[j][0]);
    drawLink("a", `L0-${j}`, W0[j][1]);
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

  // Node rects and labels
  layers.forEach((layer, li) => {
    layer.ids.forEach((id, j) => {
      const { cx, cy } = node2coord[id];
      let fillColor = "#888";
      if (layer.layerIndex !== undefined) {
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
        .text(layer.labels[j] || id);
    });
  });
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
  const margin = { top: 8, right: 52, bottom: 22, left: 48 };
  const width = w - margin.left - margin.right;
  const height = h - margin.top - margin.bottom;

  const xScale = d3.scale.linear()
    .domain([Math.max(0, iter - LOSS_HISTORY_MAX), iter])
    .range([0, width]);

  const allVals: number[] = [];
  lossHistory.forEach(d => {
    allVals.push(d.L_res, d.L_grad, d.L_data, d.L_res + d.L_grad + d.L_data);
  });
  const minV = Math.max(1e-8, d3.min(allVals)!);
  const maxV = d3.max(allVals)!;
  const yScale = d3.scale.log()
    .domain([minV, maxV])
    .range([height, 0]);

  const svg = container.append("svg").attr("width", w).attr("height", h)
    .append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  const series = [
    { key: "L_res", color: "#f59322" },
    { key: "L_grad", color: "#0877bd" },
    { key: "L_data", color: "#0a0" },
    { key: "total", color: "#333" },
  ] as const;

  series.forEach((s) => {
    const line = d3.svg.line<LossPoint>()
      .x(d => xScale(d.step))
      .y(d => {
        const v = s.key === "total" ? d.L_res + d.L_grad + d.L_data : (d as any)[s.key];
        return yScale(Math.max(minV, v));
      });
    svg.append("path")
      .datum(lossHistory)
      .attr("fill", "none")
      .attr("stroke", s.color)
      .attr("stroke-width", s.key === "total" ? 1.5 : 1)
      .attr("stroke-dasharray", s.key === "total" ? "0" : "2,2")
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

  const legend = svg.append("g").attr("transform", "translate(" + (width + 2) + ",0)");
  series.forEach((s, i) => {
    legend.append("line").attr("x1", 0).attr("y1", i * 10).attr("x2", 8).attr("y2", i * 10)
      .attr("stroke", s.color).attr("stroke-width", s.key === "total" ? 1.5 : 1);
    legend.append("text").attr("x", 10).attr("y", i * 10 + 3).attr("font-size", "8px").text(s.key);
  });
}

const HEATMAP_T_DOMAIN: [number, number] = [0, T_PLOT_MAX];
const HEATMAP_A_DOMAIN: [number, number] = [0.5, 2.5];

let plotSvg: d3.Selection<any> | null = null;

function redrawPlot() {
  const container = d3.select("#plot-1d");
  container.selectAll("*").remove();

  const node = container.node() as HTMLElement;
  const w = node.offsetWidth || 500;
  const h = node.offsetHeight || 240;
  const margin = { top: 8, right: 80, bottom: 28, left: 40 };
  const width = w - margin.left - margin.right;
  const height = h - margin.top - margin.bottom;

  const aDisplay = mode === "pretrain" ? aParam : a_learned;
  const tPlot = Array.from({ length: N_PLOT }, (_, i) => 0 + (T_PLOT_MAX - 0) * i / (N_PLOT - 1));
  const uPred = model.evalUArray(tPlot, aDisplay);
  const uExact = tPlot.map(t => Math.exp(aParamGT * t));

  let yMax = Math.max(d3.max(uPred)!, d3.max(uExact)!);
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
    .datum(uExact)
    .attr("fill", "none")
    .attr("stroke", "#333")
    .attr("stroke-width", 1.5)
    .attr("stroke-dasharray", "4,2")
    .attr("d", line);

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
  plotSvg.append("line").attr("x1", width + 8).attr("y1", legendY).attr("x2", width + 28).attr("y2", legendY)
    .attr("stroke", "#0877bd").attr("stroke-width", 2);
  plotSvg.append("text").attr("x", width + 32).attr("y", legendY + 4).attr("font-size", "11px").text("BiLO");
  plotSvg.append("line").attr("x1", width + 8).attr("y1", legendY + 18).attr("x2", width + 28).attr("y2", legendY + 18)
    .attr("stroke", "#333").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,2");
  plotSvg.append("text").attr("x", width + 32).attr("y", legendY + 22).attr("font-size", "11px").text("GT");
  if (trainingData && trainingData.t_data.length > 0) {
    plotSvg.append("circle").attr("cx", width + 18).attr("cy", legendY + 36).attr("r", 3)
      .attr("fill", "#c00").attr("stroke", "#fff").attr("stroke-width", 1);
    plotSvg.append("text").attr("x", width + 32).attr("y", legendY + 40).attr("font-size", "11px").text("Data");
  }
}

function updateUI() {
  ensureTrainingData();
  d3.select("#iter-number").text(iter);
  drawNetwork(d3.select("#network"));
  redrawPlot();
  redrawLossChart();
}

function reset() {
  iter = 0;
  a_learned = aParam;
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

function syncOptionsFromUI() {
  lr = +(d3.select("#learningRate").node() as HTMLInputElement).value || 0.02;
  lrA = +(d3.select("#lrA").node() as HTMLInputElement).value || 0.001;
  depthOption = Math.max(2, Math.min(5, +(d3.select("#depth").node() as HTMLInputElement).value || 2));
  nHidden = Math.max(2, Math.min(16, +(d3.select("#nHidden").node() as HTMLInputElement).value || 4));
  nPoints = Math.max(5, Math.min(101, +(d3.select("#nPoints").node() as HTMLInputElement).value || 21));
  wRes = +(d3.select("#wRes").node() as HTMLInputElement).value || 1;
  wGrad = +(d3.select("#wGrad").node() as HTMLInputElement).value || 0.1;
  wData = +(d3.select("#wData").node() as HTMLInputElement).value || 0.5;
  aParam = +(d3.select("#aParam").node() as HTMLInputElement).value || 1;
  aParamGT = +(d3.select("#aParamGT").node() as HTMLInputElement).value || 2;
  noiseLevel = Math.max(0, Math.min(0.5, +(d3.select("#noise").node() as HTMLInputElement).value || 0));
  mode = (d3.select("#mode").node() as HTMLSelectElement).value as "pretrain" | "finetune";
}

function init() {
  syncOptionsFromUI();
  buildModel();
  updateUI();

  d3.select("#reset-button").on("click", () => { pause(); reset(); });
  d3.select("#play-pause-button").on("click", function() {
    if (isPlaying) pause(); else play();
    d3.select(this).classed("playing", isPlaying);
  });
  d3.select("#step-button").on("click", () => { pause(); oneStep(); });

  d3.select("#learningRate").on("change", function() { syncOptionsFromUI(); });
  d3.select("#lrA").on("change", function() { syncOptionsFromUI(); });

  d3.select("#depth").on("change", function() {
    syncOptionsFromUI();
    pause();
    reset();
  });
  d3.select("#nHidden").on("change", function() {
    syncOptionsFromUI();
    pause();
    reset();
  });
  d3.select("#nPoints").on("change", function() {
    syncOptionsFromUI();
    t_colloc = getTPoints();
    a_colloc = t_colloc.map(() => (mode === "pretrain" ? aParam : a_learned));
    if (mode === "finetune") trainingData = generateTrainingData();
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
    aParam = +(this as HTMLInputElement).value || 1;
    if (mode === "pretrain") a_colloc = t_colloc.map(() => aParam);
    redrawPlot();
  });
  d3.select("#mode").on("change", function() {
    mode = (this as HTMLSelectElement).value as "pretrain" | "finetune";
    a_learned = aParam;
    a_colloc = t_colloc.map(() => (mode === "pretrain" ? aParam : a_learned));
    if (mode === "finetune") {
      trainingData = generateTrainingData();
    } else {
      trainingData = null;
    }
    redrawPlot();
  });

  window.addEventListener("resize", () => { redrawPlot(); redrawLossChart(); });
}

init();
