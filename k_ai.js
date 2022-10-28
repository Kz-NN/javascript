/**
 * A function that return the sigmoided of a number.
 * @returns The sigmoided number.
 */
 function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * A function that return the derivated of a sigmoided number.
 * @returns The derivated number.
 */
function dsigmoid(y) {
  return y * (1 - y);
}

/**
 * A class that represent a perceptron.
 */
class Perceptron {
  /**
   * Create a perceptron.
   * @param {Number} size The number of inputs.
   */
  constructor(size) {
    // Variables definition
    this.size = size;
    this.bias = Math.random() * 2 - 1;
    this.weights = [];
    for (let i = 0; i < size; i++) {
      this.weights[i] = Math.random() * 2 - 1;
    }
  }

  /**
   * Feed forward from an array of values.
   * @param {Array} inputArray The array of values to feed forward.
   * @returns The result of the feed forwarding.
   */
  getOutput(inputArray) {
    let result = 0;
    for (let i in inputArray) {
      result += this.weights[i] * inputArray[i];
    }
    result += this.bias;
    result = sigmoid(result);

    return result;
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == "string") {
      data = JSON.parse(data);
    }
    let perc = new Perceptron(data.size);
    perc.bias = data.bias;
    perc.weights = data.weights;
    return perc;
  }
}
/**
 * A class that represent a layer of perceptrons.
 */
class Layer {
  /**
   * Create a layer of perceptrons.
   * @param {Number} layerSize The number of perceptrons in the layer.
   * @param {Number} percSize The number of inputs of each perceptron.
   */
  constructor(layerSize, percSize) {
    this.layerSize = layerSize;
    this.percSize = percSize;
    this.perceptrons = [];

    for (let i = 0; i < layerSize; i++) {
      this.perceptrons.push(new Perceptron(percSize));
    }
  }

  /**
   * Feed forward from an array of values.
   * @param {Array} inputArray The array of values to feed forward.
   * @returns The result of the feed forwarding.
   */
  feedForward(inputArray) {
    if (inputArray.length !== this.percSize)
      throw new Error("Wrong number of input.");

    let result = [];
    for (const perceptron of this.perceptrons) {
      result.push(perceptron.getOutput(inputArray));
    }

    return result;
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == "string") {
      data = JSON.parse(data);
    }
    let layer = new Layer(data.layerSize, data.percSize);
    layer.perceptrons = data.perceptrons.map(Perceptron.deserialize);
    return layer;
  }
}

/**
 * A class that represent a neural network.
 */
class NeuralNetwork {
  /**
   * Create a neural nework.
   * @param {Number} iLayer The number of inputs.
   * @param {Number} hLayer The number of hidden perceptrons.
   * @param {Number} oLayer The number of outputs.
   */
  constructor(iLayer, hLayer, oLayer) {
    this.iLayer = iLayer;
    this.hLayer = new Layer(hLayer, iLayer);
    this.oLayer = new Layer(oLayer, hLayer);

    this.learningRate = 0.1;
  }

  /**
   * Predict from an inputs array.
   * @param {Array<Number} inputArray The inputs array.
   * @returns The result of the forwarding.
   */
  feedForward(inputArray) {
    let hLayerResult = this.hLayer.feedForward(inputArray);
    let oLayerResult = this.oLayer.feedForward(hLayerResult);

    return oLayerResult;
  }

  /**
   * Train the neural network from an inputs and a targets array.
   * @param {Array<Number>} inputArray The input array.
   * @param {Array<Number>} targetArray The wanted result.
   */
  train(inputArray, targetArray) {
    // Train the neural network
    let hLayerResult = this.hLayer.feedForward(inputArray);
    let oLayerResult = this.oLayer.feedForward(hLayerResult);

    // Calculate the error
    let oLayerError = [];
    for (let i in oLayerResult) {
      oLayerError[i] = targetArray[i] - oLayerResult[i];
    }

    // Calculate the delta
    let oLayerDelta = [];
    for (let i in oLayerResult) {
      oLayerDelta[i] = oLayerError[i] * dsigmoid(oLayerResult[i]);
    }

    // Calculate the error
    let hLayerError = [];
    for (let i in hLayerResult) {
      hLayerError[i] = 0;
      for (let j in oLayerDelta) {
        hLayerError[i] +=
          oLayerDelta[j] * this.oLayer.perceptrons[j].weights[i];
      }
    }

    // Calculate the delta
    let hLayerDelta = [];
    for (let i in hLayerResult) {
      hLayerDelta[i] = hLayerError[i] * dsigmoid(hLayerResult[i]);
    }

    // Update the weights
    for (let i in this.oLayer.perceptrons) {
      for (let j in hLayerResult) {
        this.oLayer.perceptrons[i].weights[j] +=
          this.learningRate * oLayerDelta[i] * hLayerResult[j];
      }
      this.oLayer.perceptrons[i].bias += this.learningRate * oLayerDelta[i];
    }

    for (let i in this.hLayer.perceptrons) {
      for (let j in inputArray) {
        this.hLayer.perceptrons[i].weights[j] +=
          this.learningRate * hLayerDelta[i] * inputArray[j];
      }
      this.hLayer.perceptrons[i].bias += this.learningRate * hLayerDelta[i];
    }
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == "string") {
      data = JSON.parse(data);
    }
    let nn = new NeuralNetwork(
      data.iLayer,
      data.hLayer.layerSize,
      data.oLayer.layerSize
    );
    nn.hLayer = Layer.deserialize(data.hLayer);
    nn.oLayer = Layer.deserialize(data.oLayer);
    return nn;
  }
}

if (typeof module === "undefined")
  module.exports = NeuralNetwork;