package lib.ai.network;

import lib.ai.neuron.Neuron;
import java.util.ArrayList;
import java.util.List;

public class Network {
    private List<Neuron[]> layers;
    private double lr;

    public Network(int neuronCount, int inputSize, double[] inputs, double lr) {
        this.layers = new ArrayList<>();
        this.lr = lr;
        Neuron[] firstLayer = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            firstLayer[i] = new Neuron(inputSize, inputs, lr);
        }
        layers.add(firstLayer);
    }

    public double predict(double target) {
        double[] currentInputs = layers.get(0)[0].getInputs();
        for (Neuron[] layer : layers) {
            double[] nextInputs = new double[layer.length];
            for (int i = 0; i < layer.length; i++) {
                layer[i].setInputs(currentInputs);
                nextInputs[i] = layer[i].predict();
            }
            currentInputs = nextInputs;
        }

        double best = currentInputs[0];
        for (double output : currentInputs) {
            if (Math.abs(output - target) < Math.abs(best - target)) {
                best = output;
            }
        }
        return best;
    }

    public synchronized void trainLayers(double[] initialInputs, double target) {
        double[] currentInput = initialInputs;
        for (int i = 0; i < layers.size(); i++) {
            Neuron[] layer = layers.get(i);
            double[] outputs = new double[layer.length];
            for (int j = 0; j < layer.length; j++) {
                layer[j].setInputs(currentInput);
                outputs[j] = layer[j].predict();
            }
            currentInput = outputs;
        }

        double[] nextLayerDeltas = null;

        for (int i = layers.size() - 1; i >= 0; i--) {
            Neuron[] currentLayer = layers.get(i);
            double[] currentDeltas = new double[currentLayer.length];

            for (int j = 0; j < currentLayer.length; j++) {
                double output = currentLayer[j].getLastOutput();
                double derivative = output * (1.0 - output);
                double error = 0;

                if (i == layers.size() - 1) {
                    error = target - output;
                } else {
                    Neuron[] nextLayer = layers.get(i + 1);
                    for (int k = 0; k < nextLayer.length; k++) {
                        error += nextLayerDeltas[k] * nextLayer[k].getWeights()[j];
                    }
                }

                double delta = error * derivative;
                currentDeltas[j] = delta;

                double[] w = currentLayer[j].getWeights();
                double[] in = currentLayer[j].getInputs();
                for (int k = 0; k < w.length; k++) {
                    w[k] += lr * delta * in[k];
                }
                currentLayer[j].setBias(currentLayer[j].getBias() + lr * delta);
            }
            nextLayerDeltas = currentDeltas;
        }
    }

    public void addLayer(int neuronCount, int inputSize, double[] inputs, double lr) {
        Neuron[] newLayer = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            newLayer[i] = new Neuron(inputSize, inputs, lr);
        }
        layers.add(newLayer);
    }

    public void setInputs(double[] inputs) {
        for (Neuron n : layers.get(0)) {
            n.setInputs(inputs);
        }
    }
}