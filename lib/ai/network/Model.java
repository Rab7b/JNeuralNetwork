package lib.ai.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Model {
    private List<double[][]> weights;
    private List<double[]> lastActivations;
    @SuppressWarnings("unused")
    private double[] currentInputs;
    private double lr;
    private double gamma = 0.95;

    public Model(int[] topology, double lr) {
        this.lr = lr;
        this.weights = new ArrayList<>();
        this.lastActivations = new ArrayList<>();
        Random rand = new Random();

        for (int i = 0; i < topology.length - 1; i++) {
            double[][] layerWeights = new double[topology[i + 1]][topology[i] + 1];
            for (int r = 0; r < layerWeights.length; r++) {
                for (int c = 0; c < layerWeights[r].length; c++) {
                    layerWeights[r][c] = (rand.nextDouble() * 2 - 1) * Math.sqrt(2.0 / topology[i]);
                }
            }
            weights.add(layerWeights);
        }
    }

    private double activation(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double[] addBias(double[] input) {
        double[] newIn = new double[input.length + 1];
        System.arraycopy(input, 0, newIn, 0, input.length);
        newIn[input.length] = 1.0;
        return newIn;
    }

    public synchronized double[] predict(double[] inputs) {
        this.currentInputs = inputs;
        lastActivations.clear();

        double[] current = addBias(inputs);
        lastActivations.add(current);

        for (int i = 0; i < weights.size(); i++) {
            double[][] layer = weights.get(i);
            double[] next = new double[layer.length];

            for (int r = 0; r < layer.length; r++) {
                double sum = 0;
                for (int c = 0; c < layer[r].length; c++) {
                    sum += current[c] * layer[r][c];
                }

                if (i < weights.size() - 1) {
                    next[r] = activation(sum);
                } else {
                    next[r] = sum;
                }
            }

            if (i < weights.size() - 1) {
                current = addBias(next);
            } else {
                current = next;
            }
            lastActivations.add(current);
        }
        return softmax(current);
    }

    private double[] softmax(double[] z) {
        double[] res = new double[z.length];
        double sum = 0;
        double max = z[0];
        for (double v : z)
            if (v > max)
                max = v;
        for (int i = 0; i < z.length; i++) {
            res[i] = Math.exp(z[i] - max);
            sum += res[i];
        }
        for (int i = 0; i < z.length; i++)
            res[i] /= sum;
        return res;
    }

    public synchronized void train(double[] inputs, double[] targets) {
        double[] output = predict(inputs);

        double[] errors = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            errors[i] = output[i] - targets[i];
        }

        // Идем назад по слоям
        for (int i = weights.size() - 1; i >= 0; i--) {
            double[][] layer = weights.get(i);
            double[] prevActivations = lastActivations.get(i);
            double[] currentActivations = lastActivations.get(i + 1);
            double[] nextErrors = new double[prevActivations.length];

            for (int r = 0; r < layer.length; r++) {

                double gradient = errors[r];
                if (i < weights.size() - 1) {
                    gradient *= currentActivations[r] * (1.0 - currentActivations[r]);
                }

                for (int c = 0; c < layer[r].length; c++) {
                    nextErrors[c] += gradient * layer[r][c];
                    layer[r][c] -= lr * gradient * prevActivations[c];
                }
            }

    
            if (i > 0) {
                double[] errorsForNext = new double[nextErrors.length - 1];
                System.arraycopy(nextErrors, 0, errorsForNext, 0, errorsForNext.length);
                errors = errorsForNext;
            }
        }
    }

    public synchronized void qLearn(double[] state, int action, double reward, double[] nextState) {
        double[] currentQs = predict(state);
        double[] nextQs = predict(nextState);

        double maxNextQ = nextQs[0];
        for (double q : nextQs)
            if (q > maxNextQ)
                maxNextQ = q;

        double targetQ = reward + (gamma * maxNextQ);

        double[] targetVector = currentQs.clone();
        targetVector[action] = targetQ;

        train(state, targetVector);
    }

    public void setInputs(double[] inputs) {
        this.currentInputs = inputs;
    }

    public List<double[][]> getWeight() {
        return this.weights;
    }

    public void setWeight(List<double[][]> newWeights) {
        if (newWeights != null && !newWeights.isEmpty()) {
            this.weights = newWeights;
        }
    }

    public List<double[][]> getWeights() {
        return getWeight();
    }

    public double[] getWeightsFlat() {
        List<double[][]> allWeights = getWeight();
        int totalSize = 0;
        for (double[][] matrix : allWeights) {
            totalSize += matrix.length * matrix[0].length;
        }

        double[] flat = new double[totalSize];
        int offset = 0;
        for (double[][] matrix : allWeights) {
            for (double[] row : matrix) {
                for (double val : row) {
                    flat[offset++] = val;
                }
            }
        }
        return flat;
    }

    public void setWeightsFlat(double[] flat) {
        int offset = 0;
        for (double[][] matrix : weights) {
            for (int r = 0; r < matrix.length; r++) {
                for (int c = 0; c < matrix[r].length; c++) {
                    matrix[r][c] = flat[offset++];
                }
            }
        }
    }
}