export class Activation {
  func;
  dfunc;

  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

export const SIGMOID = new Activation(
  (x) => 1 / (1 + Math.E ** -x),
  (y) => y * (1 - y)
);

export const TANH = new Activation(
  (x) => Math.tanh(x),
  (y) => 1 - y ** 2
);
