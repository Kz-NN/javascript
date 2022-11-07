/****************************
 *      KAI (c) Kelbaz      *
 *      "`"`"`"`"`"`"`      *
/***************************/

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
     * @param {Array<Number>} hLayers The number of hidden perceptrons.
     * @param {Number} oLayer The number of outputs.
     */
    constructor(iLayer, hLayers, oLayer) {
        this.iLayer = iLayer;
        this.hLayers = [];
        let prevLayer = iLayer;
        for (const layer of hLayers) {
            this.hLayers.push(new Layer(layer, prevLayer));
            prevLayer = layer;
        }
        this.oLayer = new Layer(oLayer, prevLayer);

        this.learningRate = 0.1;
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
     * Train the neural network from an inputs and a targets array.
     * @param {Array<Number>} inputArray The input array.
     * @param {Array<Number>} targetArray The wanted result.
     */
    train(inputArray, targetArray) {
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
            oLayerDelta[i] = oLayerError[i] * dsigmoid(result[i]);
        }

        // Calculate the error
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

        // Calculate the delta
        let hLayersDelta = [];
        for (let i = this.hLayers.length - 1; i >= 0; i--) {
            hLayersDelta[i] = [];
            for (let j in this.hLayers[i].perceptrons) {
                hLayersDelta[i][j] =
                    hLayersError[i][j] * dsigmoid(results[i + 1][j]);
            }
        }

        // Update the weights
        for (let i in this.oLayer.perceptrons) {
            for (let j in results[this.hLayers.length]) {
                this.oLayer.perceptrons[i].weights[j] +=
                    this.learningRate *
                    oLayerDelta[i] *
                    results[this.hLayers.length][j];
            }
            this.oLayer.perceptrons[i].bias +=
                this.learningRate * oLayerDelta[i];
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
     * Train the neural network from a training data a particular amount of time.
     * @param {Array<Object>} trainingData The training data.
     * @param {Number} iterations The number of iteration.
     */
    trainFromData(trainingData, iterations) {
        for (let i = 0; i < iterations; i++) {
            for (const data of trainingData) {
                this.train(data.input, data.target);
            }
        }
    }

    /**
     * Train the neural network from a training data until the error is smaller than a particular value.
     * @param {Array<Object>} trainingData The training data.
     * @param {Number} error The error.
     */
    trainUntilError(trainingData, error) {
        let iteration = 0;
        let lastError = 0;
        while (Math.abs(lastError) <= error) {
            lastError = 0;
            iteration++;
            for (const data of trainingData) {
                this.train(data.input, data.target);
                lastError += Math.abs(
                    this.feedForward(data.input)[0] - data.target[0]
                );
            }
            lastError /= trainingData.length;
        }
        return iteration;
    }

    /**
     * Serialize the neural network
     * @returns The serialized neural network.
     */
    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(data) {
        if (typeof data == "string") {
            data = JSON.parse(data);
        }
        let nn = new NeuralNetwork(
            data.iLayer,
            data.hLayers.map((layer) => layer.layerSize),
            data.oLayer.layerSize
        );
        nn.hLayers = data.hLayers.map(Layer.deserialize);
        nn.oLayer = Layer.deserialize(data.oLayer);
        return nn;
    }
}

if (typeof module !== "undefined") module.exports = NeuralNetwork;
