/**
 * BILO training: pretrain (fix a=1) and optional finetune with data
 */

import { BILOModel } from "./bilo_nn";

export interface TrainStep {
  step: number;
  L_res: number;
  L_grad: number;
  L_data: number;
  L_total: number;
  a?: number;
}

function addVec(a: number[], b: number[], scale: number): number[] {
  return a.map((x, i) => x - scale * b[i]);
}

export function train(
  model: BILOModel,
  t_colloc: number[],
  a_colloc: number[],
  opts: { n_iters: number; lr: number; w_res?: number; w_grad?: number; log_every?: number }
): TrainStep[] {
  const { n_iters, lr, w_res = 1, w_grad = 0.1, log_every = 100 } = opts;
  const history: TrainStep[] = [];
  for (let step = 0; step < n_iters; step++) {
    const { losses, grads } = model.computeLossesAndGradients(t_colloc, a_colloc, { w_res, w_grad, w_data: 0 });
    const L_total = losses.L_res + losses.L_grad + losses.L_data;

    for (let k = 0; k < model.depth; k++) {
      const Wk = model._W[k];
      const gWk = grads[`W${k + 1}`];
      if (Array.isArray(Wk[0])) {
        const W = Wk as number[][];
        const g = gWk as number[][];
        for (let i = 0; i < W.length; i++)
          for (let j = 0; j < W[i].length; j++) W[i][j] -= lr * g[i][j];
      } else {
        const W = Wk as number[];
        const g = gWk as number[];
        for (let i = 0; i < W.length; i++) W[i] -= lr * g[i];
      }
      const bk = model._b[k];
      const gbk = grads[`b${k + 1}`];
      if (typeof bk === "number") {
        (model._b[k] as number) = bk - lr * (gbk as number);
      } else {
        const b = bk as number[];
        const g = gbk as number[];
        for (let i = 0; i < b.length; i++) b[i] -= lr * g[i];
      }
    }

    history.push({ step, ...losses, L_total });
  }
  return history;
}
