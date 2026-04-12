package lib.ai.network;

import lib.ai.neuron.Neuron;

public class Network {
    private Neuron[] neurons;

    public Network(int neuronCount, int inputSize, double[] inputs, double lr) {
        neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputSize, inputs, lr);
        }
    }

    public double predict(double target) {
        double best = 0.0;
        for (int i = 0; i < neurons.length; i++) {
            double output = neurons[i].predict();
            if (Math.abs(output - target) < Math.abs(best - target)) {
                best = output;
            }
        }
        return best;
    }

    public void train(double target) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].train(target);
        }
    }

    public void qlearn(double reward, double next, double gamma) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].qlearn(reward, next, gamma);
        }
    }

    public void addLayer(int neuronCount, int inputSize, double[] inputs, double lr) {
        Neuron[] newNeurons = new Neuron[neurons.length + neuronCount];
        for (int i = 0; i < neurons.length; i++) {
            newNeurons[i] = neurons[i];
        }
        for (int i = neurons.length; i < newNeurons.length; i++) {
            newNeurons[i] = new Neuron(inputSize, inputs, lr);
        }
        neurons = newNeurons;
    }

    public void setInputs(double[] inputs) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].setInputs(inputs);
        }
    }
}
