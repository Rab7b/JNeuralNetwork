package ai.neuron;

public class Neuron {
    private double[] weights, inputs;
    private double bias;
    private double lr;
    public Neuron(int inputSize, double[] in, double lr) {
        this.lr = lr;
        weights = new double[inputSize];
        inputs = new double[inputSize];
        inputs = in;
        for (int i = 0; i < inputSize; i++) {
            weights[i] = Math.random() * 2 - 1;
        }
        bias = 0;
    }
    public double activate(double x){
        return Math.max(0.01*x, x);
    }
    public double derivative(double x){
        return x > 0 ? 1 : 0.01;
    }
    public double predict() {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return activate(sum);
    }
    public void train(double target) {
        double output = predict();
        double error = target - output;
        for (int i = 0; i < weights.length; i++) {
            weights[i] += lr * error * inputs[i] * derivative(output);
        }
        bias += lr * error * derivative(output);
    }
    public void qlearn(double reward, double next, double gamma) {
        double output = predict();
        double target = reward + gamma * next;
        double error = target - output;
        for (int i = 0; i < weights.length; i++) {
            weights[i] += lr * error * inputs[i] * derivative(output);
        }
        bias += lr * error * derivative(output);
    }
}