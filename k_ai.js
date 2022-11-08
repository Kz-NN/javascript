/****************************
 *      KAI (c) Kelbaz      *
 * v1.4 `"`"`"`"`"`"`"      *
/***************************/

const KAI = {};

KAI.ActivationFunction = class {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
};

KAI.sigmoid = new KAI.ActivationFunction(
  (x) => 1 / (1 + Math.exp(-x)),
  (y) => y * (1 - y)
);

KAI.tanh = new KAI.ActivationFunction(
  (x) => Math.tanh(x),
  (y) => 1 - y * y
);

/**
 * A class that represent a perceptron.
 */
KAI.Perceptron = class {
  /**
   * Create a perceptron.
   * @param {Number} size The number of inputs.
   * @param {KAI.ActivationFunction} activationFunction The activation function to use.
   */
  constructor(size, activationFunction = KAI.sigmoid) {
    // Variables definition
    this.size = size;
    this.activation = activationFunction;
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
    result = this.activation.func(result);

    return result;
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == "string") {
      data = JSON.parse(data);
    }
    let perc = new KAI.Perceptron(data.size);
    perc.bias = data.bias;
    perc.weights = data.weights;
    return perc;
  }
};
/**
 * A class that represent a layer of perceptrons.
 */
KAI.Layer = class {
  /**
   * Create a layer of perceptrons.
   * @param {Number} layerSize The number of perceptrons in the layer.
   * @param {Number} percSize The number of inputs of each perceptron.
   * @param {KAI.ActivationFunction} activationFunction The activation function to use.
   */
  constructor(layerSize, percSize, activationFunction = KAI.sigmoid) {
    this.layerSize = layerSize;
    this.percSize = percSize;
    this.activation = activationFunction;
    this.perceptrons = [];

    for (let i = 0; i < layerSize; i++) {
      this.perceptrons.push(new KAI.Perceptron(percSize));
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
    let layer = new KAI.Layer(data.layerSize, data.percSize);
    layer.perceptrons = data.perceptrons.map(Perceptron.deserialize);
    return layer;
  }
};

/**
 * A class that represent a neural network.
 */
KAI.NeuralNetwork = class {
  /**
   * Create a neural nework.
   * @param {Object} params The object parameters of the neural network.
   */
  constructor(params) {
    this.iLayer = params.inputLayer;
    this.hLayers = [];
    let prevLayer = this.iLayer;
    for (const layer of params.hiddenLayers) {
      this.hLayers.push(new KAI.Layer(layer, prevLayer));
      prevLayer = layer;
    }
    this.oLayer = new KAI.Layer(params.outputLayer, prevLayer);

    this.learningRate = params.learningRate || 0.1;
    this.activation = params.activationFunction || KAI.sigmoid;
  }

  /**
   * Predict from an inputs array.
   * @param {Array<Number} inputArray The inputs array.
   * @returns The result of the forwarding.
   */
  feedForward(inputArray) {
    let result = inputArray;
    for (const layer of this.hLayers) {
      result = layer.feedForward(result);
    }
    result = this.oLayer.feedForward(result);

    return result;
  }

  /**
   * Train the neural network with specific settings.
   * @param {Object} params The parameters object.
   */
  train(params) {
    let data = params.data;
    // if there is a "iterations" parameter we train the datas this number of times
    if (params.iterations) {
      for (let i = 0; i < params.iterations; i++) {
        for (const d of data) {
          this.trainOne(d.input, d.target);
        }
      }
    }
    // if there is a "error" parameter we train the datas until the error is smaller than this value
    else if (params.error) {
      let iteration = 0;
      let lastError = 0;
      while (Math.abs(lastError) <= params.error) {
        lastError = 0;
        iteration++;
        for (const d of data) {
          this.trainOne(d.input, d.target);
          lastError += Math.abs(this.feedForward(d.input)[0] - d.target[0]);
        }
        lastError /= data.length;
      }
      return iteration;
    } else {
      for (let i = 0; i < params.iterations; i++) {
        for (const d of data) {
          this.trainOne(d.input, d.target);
        }
      }
    }
  }

  /**
   * Train the neural network from an inputs and a targets array.
   * @param {Array<Number>} inputArray The input array.
   * @param {Array<Number>} targetArray The wanted result.
   */
  trainOne(inputArray, targetArray) {
    // Train the neural network
    let result = inputArray;
    let results = [result];
    for (const layer of this.hLayers) {
      result = layer.feedForward(result);
      results.push(result);
    }
    result = this.oLayer.feedForward(result);
    results.push(result);

    // Calculate the error
    let oLayerError = [];
    for (let i in result) {
      oLayerError[i] = targetArray[i] - result[i];
    }

    // Calculate the delta
    let oLayerDelta = [];
    for (let i in result) {
      oLayerDelta[i] = oLayerError[i] * this.activation.dfunc(result[i]);
    }

    // Calculate the hidden layers error
    let hLayersError = [];
    for (let i = this.hLayers.length - 1; i >= 0; i--) {
      hLayersError[i] = [];
      for (let j in this.hLayers[i].perceptrons) {
        hLayersError[i][j] = 0;
        for (let k in oLayerDelta) {
          hLayersError[i][j] +=
            oLayerDelta[k] * this.oLayer.perceptrons[k].weights[j];
        }
      }
    }

    // Calculate the hidden layers delta
    let hLayersDelta = [];
    for (let i = this.hLayers.length - 1; i >= 0; i--) {
      hLayersDelta[i] = [];
      for (let j in this.hLayers[i].perceptrons) {
        hLayersDelta[i][j] =
          hLayersError[i][j] * this.activation.dfunc(results[i + 1][j]);
      }
    }

    // Update the weights
    for (let i in this.oLayer.perceptrons) {
      for (let j in results[this.hLayers.length]) {
        this.oLayer.perceptrons[i].weights[j] +=
          this.learningRate * oLayerDelta[i] * results[this.hLayers.length][j];
      }
      this.oLayer.perceptrons[i].bias += this.learningRate * oLayerDelta[i];
    }

    for (let i = this.hLayers.length - 1; i >= 0; i--) {
      for (let j in this.hLayers[i].perceptrons) {
        for (let k in results[i]) {
          this.hLayers[i].perceptrons[j].weights[k] +=
            this.learningRate * hLayersDelta[i][j] * results[i][k];
        }
        this.hLayers[i].perceptrons[j].bias +=
          this.learningRate * hLayersDelta[i][j];
      }
    }
  }

  /**
   * Serialize the neural network
   * @returns The serialized neural network.
   */
  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == "string") data = JSON.parse(data);

    let nn = new KAI.NeuralNetwork({
      inputLayer: data.iLayer,
      hiddenLayers: data.hLayers.map((layer) => layer.layerSize),
      outputLayer: data.oLayer.layerSize,
    });

    nn.hLayers = data.hLayers.map(Layer.deserialize);
    nn.oLayer = Layer.deserialize(data.oLayer);
    return nn;
  }
};

if (typeof module !== "undefined") module.exports = KAI;
