import lib.ai.network.*;
import lib.support.*;

public class Main {
    public static void main(String[] args) {

        MultiTask task = new MultiTask();

        Network network = new Network(4, 2, new double[] { 0, 0 }, 0.1);

        network.addLayer(2, 2, new double[] { 0, 0 }, 0.1);
        network.addLayer(2, 2, new double[] { 0, 0 }, 0.1);

        double[][] inputs = {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
        };
        double[] targets = { 0, 1, 1, 0 };

        for (int epoch = 0; epoch < 100000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                final int index = i;
                network.setInputs(inputs[index]);
                task.doIt((progress) -> {
                    network.trainLayers(inputs[index],targets[index]);
                    return null;
                });
            }
        }
        task.waitForCompletion();
        for (int i = 0; i < inputs.length; i++) {
            network.setInputs(inputs[i]);
            double output = network.predict(targets[i]);
            System.out.println("Input: " + inputs[i][0] + ", " + inputs[i][1] + " | Output: "
                    + String.format("%.4f", output) + " | Target: " + targets[i]); 
        }
    }
}
