package lib.ai.neuron;
import lib.support.*;

public class Neuron {
    private double[] weights, inputs;
    private double bias;
    private double lr;
    private double lastOutput;
    Activation activation;

    public Neuron(int inputSize, double[] in, double lr, Activation activation) {
        this.lr = lr;
        this.weights = new double[inputSize];
        this.inputs = (in != null) ? in : new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            this.weights[i] = Math.random() * 2 - 1;
        }
        this.bias = Math.random() * 2 - 1;
        this.activation = activation;
    }

    public double predict() {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        lastOutput = activation.forward(sum);
        return lastOutput;
    }

    public synchronized void train(double target) {
        double output = predict();
        double error = target - output;
        double delta = error * activation.derivative(output);
        for (int i = 0; i < weights.length; i++) {
            weights[i] += lr * delta * inputs[i];
        }
        bias += lr * delta;
    }

    public synchronized void qlearn(double reward, double next, double gamma) {
        double output = predict();
        double target = reward + gamma * next;
        double error = target - output;
        double delta = error * activation.derivative(output);
        for (int i = 0; i < weights.length; i++) {
            weights[i] += lr * delta * inputs[i];
        }
        bias += lr * delta;
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    public double[] getInputs() {
        return inputs;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights){
        this.weights = weights;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getLastOutput() {
        return lastOutput;
    }
}
