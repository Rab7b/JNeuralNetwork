package lib.ai.neuron;

public class Neuron {
    private double[] weights, inputs;
    private double bias;
    private double lr;
    private double lastOutput;

    public Neuron(int inputSize, double[] in, double lr) {
        this.lr = lr;
        this.weights = new double[inputSize];
        this.inputs = (in != null) ? in : new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            this.weights[i] = Math.random() * 2 - 1;
        }
        this.bias = Math.random() * 2 - 1;
    }

    private static double activate(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private static double derivative(double x) {
        return x * (1.0 - x);
    }

    public double predict() {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        lastOutput = activate(sum);
        return lastOutput;
    }

    public void train(double target) {
        double output = predict();
        double error = target - output;
        double delta = error * derivative(output);
        for (int i = 0; i < weights.length; i++) {
            weights[i] += lr * delta * inputs[i];
        }
        bias += lr * delta;
    }

    public void qlearn(double reward, double next, double gamma) {
        double output = predict();
        double target = reward + gamma * next;
        double error = target - output;
        double delta = error * derivative(output);
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
