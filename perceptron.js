const stepFunction = (x) => {
  return x >= 0 ? 1 : 0;
};

const dot = (a, weights) => {
  let sum = 0;
  for (let i = 0; i < weights.length; i++) {
    sum += a[i] * weights[i];
  }
  return sum;
};

const randomArray = (length) => {
  return Array.from({ length }, () => Math.random());
};

class Neuron {
  weights;
  bias;
  learningRate;

  constructor(inputSize, learningRate = 0.3) {
    this.weights = randomArray(inputSize);
    this.bias = Math.random();
    this.learningRate = learningRate;
  }

  predict(x) {
    const z = dot(x, this.weights) + this.bias;
    // console.log(`Z: ${z}`);
    return stepFunction(z);
  }

  train(x, y, epochs = 10) {
    // console.log("Treinando...");
    for (let i = 0; i < epochs; i++) {
      x.forEach((xi, index) => {
        const target = y[index];
        const output = this.predict(xi);
        const error = target - output;

        this.weights = this.weights.map(
          (weight, weightIndex) =>
            weight + this.learningRate * error * xi[weightIndex]
        );
        this.bias += this.learningRate * error;
      });
    }
  }
}

const X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

const Y = [0, 0, 0, 1];
const repetitions = 1000;

const training = (actualRep = 0, weights = [], bias = 0, oldNeuron = null) => {
  if (actualRep === repetitions) {
    console.log("Repeticoes concluidas");
    for (let i = 0; i < X.length; i++) {
      console.log(`${X[i]} -> ${oldNeuron.predict(X[i])}`);
    }
    return;
  }

  const neuron = new Neuron(2);
  if (weights.length > 0) {
    neuron.weights = weights;
  }
  if (bias !== 0) {
    neuron.bias = bias;
  }
  neuron.train(X, Y, 10);
  training(actualRep + 1, neuron.weights, neuron.bias, neuron);
};

training();
