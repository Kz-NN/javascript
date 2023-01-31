import Matrix from "./matrix.js";

export class Network {
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
        console.log(`Iteration ${i} of ${iter} (${Math.round((i / iter) * 100)}%)`);
      }

      for (let j = 0; j < dataset.length; j++) {
        let outputs = this.feedForward(dataset[j].inputs);
        this.backPropagate(outputs, dataset[j].targets);
      }
    }
  }
}
