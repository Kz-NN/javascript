/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
* v1.6                        Kelbaz's Neural Network                 By Kelbaz *
*                                                                               *
*        @@@@@@@@@@@@@@                    @@,      @@@@@   @@,      @@@@@      *
*        @@@@@@@@@@@@'    @@@@@@@@@@@@@@   @@@@,    @@@@@   @@@@,    @@@@@      *
*     ,@@ `@@@@@@@@'      @@@@@@@@@@@@'    @@@@@@,  @@@@@   @@@@@@,  @@@@@      *
*     @@@  :@@@@@'           ,@@@@@@'      @@@@@@@@,@@@@@   @@@@@@@@,@@@@@      *
*     `@@ ,@@@@' @@,       ,@@@@@@'        @@@@@@@@@ @@@@   @@@@@@@@@ @@@@      *
*        @@@@'   @@@@,         ,@@@@@@@@   @@@@@@@@@  "@@   @@@@@@@@@  "@@      *
*        @@'     @@@@@@      ,@@@@@@@@@@   @@@@@@@@@   `@   @@@@@@@@@   `@      *
*                                                                               *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
  Activation
*/
class Activation {
  func;
  dfunc;

  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

const SIGMOID = new Activation(
  (x) => 1 / (1 + Math.E ** -x),
  (y) => y * (1 - y)
);

const TANH = new Activation(
  (x) => Math.tanh(x),
  (y) => 1 - y ** 2
);

/*
  Matrix
*/
class Matrix {
  rows;
  cols;
  data;

  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = new Array(rows).fill().map(() => Array(cols).fill(null));
  }

  static zeros(rows, cols) {
    let m = new Matrix(rows, cols);
    m.data = new Array(rows).fill().map(() => Array(cols).fill(0));

    return m;
  }

  static random(rows, cols) {
    let m = new Matrix(rows, cols);
    m.data = new Array(rows)
      .fill()
      .map(() => Array(cols).fill(Math.random() * 2 - 1));

    return m;
  }

  static from(data) {
    let m = new Matrix(data.length, data[0].length);
    m.data = data;

    return m;
  }

  static desirialize(serialized) {
    let data = JSON.parse(serialized);
    let m = new Matrix(data.rows, data.cols);

    m.data = data.data;

    return m;
  }

  serialized() {
    return JSON.stringify(this);
  }

  multiply(other) {
    if (this.cols !== other.rows)
      throw new Error(
        "Attempted to multiply by matrix of incorrect dimensions"
      );

    let res = Matrix.zeros(this.rows, other.cols);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < other.cols; j++) {
        let sum = 0.0;
        for (let k = 0; k < this.cols; k++) {
          sum += this.data[i][k] * other.data[k][j];
        }

        res.data[i][j] = sum;
      }
    }

    return res;
  }

  add(other) {
    if (this.rows !== other.rows || this.cols !== other.cols)
      throw new Error("Attempted to add matrix of incorrect dimensions");

    let res = Matrix.zeros(this.rows, this.cols);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        res.data[i][j] = this.data[i][j] + other.data[i][j];
      }
    }

    return res;
  }

  dotMultiply(other) {
    if (this.rows !== other.rows || this.cols !== other.cols)
      throw new Error(
        "Attempted to dot multiply matrix of incorrect dimensions"
      );

    let res = Matrix.zeros(this.rows, this.cols);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        res.data[i][j] = this.data[i][j] * other.data[i][j];
      }
    }

    return res;
  }

  subtract(other) {
    if (this.rows !== other.rows || this.cols !== other.cols)
      throw new Error("Attempted to subtract matrix of incorrect dimensions");

    let res = Matrix.zeros(this.rows, this.cols);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        res.data[i][j] = this.data[i][j] - other.data[i][j];
      }
    }

    return res;
  }

  map(func) {
    return Matrix.from(
      this.data.slice().map((row) => row.map((value) => func(value)))
    );
  }

  transpose() {
    let res = Matrix.zeros(this.cols, this.rows);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        res.data[j][i] = this.data[i][j];
      }
    }

    return res;
  }
}

/*
  Network
*/
class Network {
  layers;
  weights;
  biases;
  data = [];
  learningRate;
  activation;

  constructor(layers, learningRate, activation) {
    let weights = [];
    let biases = [];

    for (let i = 0; i < layers.length - 1; i++) {
      weights.push(Matrix.random(layers[i + 1], layers[i]));
      biases.push(Matrix.random(layers[i + 1], 1));
    }

    this.layers = layers;
    this.weights = weights;
    this.biases = biases;
    this.activation = activation;
    this.learningRate = learningRate;
  }

  save() {
    return JSON.stringify({
      inputs: this.layers[0],
      weights: this.weights.map((matrix) => matrix.data),
      biases: this.biases.map((matrix) => matrix.data),
      learning_rate: this.learningRate,
    });
  }

  static deserialize(data, activation) {
    let saveData = JSON.parse(data);

    let weights = [];
    let biases = [];
    let layers = [saveData.inputs];

    for (let i = 0; i < saveData.weights.length; i++) {
      layers.push(saveData.weights[i].length);

      weights.push(Matrix.from(saveData.weights[i]));
      biases.push(Matrix.from(saveData.biases[i]));
    }

    let nn = new Network(layers, saveData.learning_rate, activation);
    nn.weights = weights;
    nn.biases = biases;

    return nn;
  }

  feedForward(inputs) {
    if (inputs.length != this.layers[0])
      throw Error("Invalid number of inputs");

    let current = Matrix.from([inputs]).transpose();
    this.data = [current];

    for (let i = 0; i < this.layers.length - 1; i++) {
      current = this.weights[i]
        .multiply(current)
        .add(this.biases[i])
        .map(this.activation.func);
      this.data.push(current);
    }

    return current.transpose().data[0];
  }

  backPropagate(outputs, targets) {
    if (targets.length !== this.layers[this.layers.length - 1]) {
      throw Error("Invalid number of targets");
    }

    let parsed = Matrix.from([outputs]).transpose();
    let errors = Matrix.from([targets]).transpose().subtract(parsed);
    let gradients = parsed.map(this.activation.dfunc);

    for (let i = this.layers.length - 2; i >= 0; i--) {
      gradients = gradients
        .dotMultiply(errors)
        .map((x) => x * this.learningRate);

      this.weights[i] = this.weights[i].add(
        gradients.multiply(this.data[i].transpose())
      );
      this.biases[i] = this.biases[i].add(gradients);

      errors = this.weights[i].transpose().multiply(errors);
      gradients = this.data[i].map(this.activation.dfunc);
    }
  }

  train(dataset, iter) {
    for (let i = 1; i <= iter; i++) {
      if (iter < 100 || i % (iter / 100) === 0) {
        console.log(
          `Iteration ${i} of ${iter} (${Math.round((i / iter) * 100)}%)`
        );
      }

      for (let j = 0; j < dataset.length; j++) {
        let outputs = this.feedForward(dataset[j].inputs);
        this.backPropagate(outputs, dataset[j].targets);
      }
    }
  }
}

module.exports = {
  activation: {
    Activation,
    SIGMOID,
    TANH,
  },
  network: { Network },
};
