import lib.ai.network.*;

public class Main {
    public static void main(String[] args) {
        Network network = new Network(1, 2, null, 0.01);
        double[][] inputs = {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
        };
        double[] targets = { 0, 1, 1, 1 };

        network.addLayer(10, 2, inputs[0], 0.01);
        network.addLayer(10, 2, inputs[1], 0.01);
        network.addLayer(10, 2, inputs[2], 0.01);
        network.addLayer(10, 2, inputs[3], 0.01);

        for (int epoch = 0; epoch < 10000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                final int index = i;
                network.setInputs(inputs[i]);
                network.train(targets[index]);
            }
        }
        for (int i = 0; i < inputs.length; i++) {
            network.setInputs(inputs[i]);
            double output = network.predict(targets[i]);
            System.out.println( "Input: " + inputs[i][0] + ", " + inputs[i][1] + " Output: " + output + " Target: " + targets[i]);
        }
    }
}
