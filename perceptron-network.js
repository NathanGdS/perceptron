const stepFunction = (x) => {
  return x >= 0 ? 1 : 0;
};

const sigmoid = (x) => {
  return 1 / (1 + Math.exp(-x));
};

const sigmoidDerivative = (x) => {
  const s = sigmoid(x);
  return s * (1 - s);
};

const dot = (a, weights) => {
  let sum = 0;
  for (let i = 0; i < weights.length; i++) {
    sum += a[i] * weights[i];
  }
  return sum;
};

const randomArray = (length) => {
  return Array.from({ length }, () => Math.random() * 2 - 1); // Pesos entre -1 e 1
};

class Neuron {
  weights;
  bias;
  learningRate;
  lastInput;
  lastOutput;
  lastZ;

  constructor(inputSize, learningRate = 0.5) {
    this.weights = randomArray(inputSize);
    this.bias = Math.random() * 2 - 1;
    this.learningRate = learningRate;
  }

  // Método para obter saída sem função de ativação (para backpropagation)
  getRawOutput(input) {
    this.lastInput = input;
    this.lastZ = dot(input, this.weights) + this.bias;
    return this.lastZ;
  }

  // Método para obter saída com função de ativação
  predict(input) {
    const z = this.getRawOutput(input);
    this.lastOutput = sigmoid(z);
    return this.lastOutput;
  }

  // Método para obter saída com step function (para compatibilidade)
  predictStep(input) {
    const z = this.getRawOutput(input);
    this.lastOutput = stepFunction(z);
    return this.lastOutput;
  }

  // Método para treinamento individual (perceptron simples)
  train(x, y, epochs = 10) {
    for (let i = 0; i < epochs; i++) {
      x.forEach((xi, index) => {
        const target = y[index];
        const output = this.predictStep(xi);
        const error = target - output;

        this.weights = this.weights.map(
          (weight, weightIndex) =>
            weight + this.learningRate * error * xi[weightIndex]
        );
        this.bias += this.learningRate * error;
      });
    }
  }

  // Método para atualizar pesos durante backpropagation
  updateWeights(delta) {
    this.weights = this.weights.map(
      (weight, index) =>
        weight + this.learningRate * delta * this.lastInput[index]
    );
    this.bias += this.learningRate * delta;
  }
}

class NeuralNetwork {
  hiddenLayer;
  outputLayer;
  learningRate;

  constructor(learningRate = 0.5) {
    this.learningRate = learningRate;
    this.hiddenLayer = [
      new Neuron(2, learningRate),
      new Neuron(2, learningRate),
    ];
    this.outputLayer = new Neuron(2, learningRate);
  }

  feedWithWeights(hiddenLayerWeights, outputLayerWeights) {
    console.log("Pesos da camada oculta:", hiddenLayerWeights);
    console.log("Pesos da camada de saída:", outputLayerWeights);
    this.hiddenLayer.forEach((neuron, index) => {
      neuron.weights = hiddenLayerWeights[index];
    });
    this.outputLayer.weights = outputLayerWeights;
  }

  // Forward pass - propagação para frente
  forward(input) {
    // Calcular saídas da camada oculta
    const hiddenOutputs = this.hiddenLayer.map((neuron) =>
      neuron.predict(input)
    );

    // Calcular saída da camada de saída
    const finalOutput = this.outputLayer.predict(hiddenOutputs);

    return {
      hiddenOutputs,
      finalOutput,
    };
  }

  // Backpropagation - propagação do erro para trás
  backward(input, target) {
    const { hiddenOutputs, finalOutput } = this.forward(input);

    // Calcular erro na camada de saída
    const outputError = target - finalOutput;
    const outputDelta = outputError * sigmoidDerivative(this.outputLayer.lastZ);

    // Calcular erros na camada oculta
    const hiddenErrors = this.hiddenLayer.map((neuron, index) => {
      return this.outputLayer.weights[index] * outputDelta;
    });

    const hiddenDeltas = hiddenErrors.map((error, index) => {
      return error * sigmoidDerivative(this.hiddenLayer[index].lastZ);
    });

    // Atualizar pesos da camada de saída
    this.outputLayer.updateWeights(outputDelta);

    // Atualizar pesos da camada oculta
    this.hiddenLayer.forEach((neuron, index) => {
      neuron.updateWeights(hiddenDeltas[index]);
    });

    return outputError;
  }

  // Treinamento da rede completa
  train(trainingData, epochs = 1000) {
    console.log("Iniciando treinamento da rede neural...");

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalError = 0;

      trainingData.forEach(({ input, target }) => {
        const error = this.backward(input, target);
        totalError += Math.abs(error);
      });

      // // Log de progresso a cada 100 épocas
      // if (epoch % 10_000 === 0) {
      //   console.log(
      //     `Época ${epoch}, Erro médio: ${(
      //       totalError / trainingData.length
      //     ).toFixed(4)}`
      //   );
      // }

      // Parar se convergiu
      if (totalError < 0.01) {
        console.log(
          `Convergência atingida na época ${epoch} com erro ${totalError}`
        );
        break;
      }
    }

    console.log("Treinamento concluído!");
  }

  // Predição final
  predict(input) {
    const { finalOutput } = this.forward(input);
    return stepFunction(finalOutput - 0.5); // Threshold para classificação binária
  }

  // Testar a rede
  test(testData) {
    console.log("\nTestando a rede neural:");
    console.log("Entrada -> Saída (Esperado)");
    console.log("------------------------");

    testData.forEach(({ input, target }) => {
      const prediction = this.predict(input);
      console.log(`${input} -> ${prediction} (${target})`);
    });
  }
}

// Dados de treinamento para função de igualdade (XNOR)
const trainingData = [
  { input: [0, 0], target: 0 }, // 0 == 0 -> 1
  { input: [0, 5], target: 1 }, // 0 != 1 -> 0
  { input: [1, 0], target: 0 }, // 1 != 0 -> 0
  { input: [5, 1], target: 0 }, // 1 == 1 -> 1
];

// Dados de teste adicionais
const testData = [
  { input: [0, 0], target: 0 },
  { input: [0, 1], target: 1 },
  { input: [1, 0], target: 1 },
  { input: [1, 1], target: 0 },
  { input: [0.5, 0.5], target: 0 }, // Teste com decimais
  { input: [0.3, 0.7], target: 1 },
];

// Função principal
const trainModel = () => {
  console.log("=== Rede Neural para Detecção de Igualdade ===\n");

  // Criar e treinar a rede
  const network = new NeuralNetwork(0.8);
  network.train(trainingData, 200_000);

  // Testar a rede
  network.test(testData);

  console.log("\n=== Alimentando a rede com pesos ===");
  network.feedWithWeights(
    network.hiddenLayer.map((neuron) => neuron.weights),
    network.outputLayer.weights
  );
  network.test(testData);
};

const modelTrained = () => {
  const network = new NeuralNetwork(0.8);
  network.feedWithWeights(
    [
      [-6.827663800118968, -6.827208942014659],
      [8.445954624004425, 8.443841207820757],
    ],
    [[12.87504120599525, 12.817182918831936]]
  );

  network.test(testData);
};

// trainModel();
modelTrained();
