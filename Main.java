import lib.ai.network.*;
import lib.ai.neuron.Savings;
import lib.support.*;
import java.io.File;

public class Main {
    public static void main(String[] args) throws Exception {

        MultiTask task = new MultiTask();

        Network network = new Network(4, 2, new double[] { 0, 0 }, 0.1);
        network.addLayer(2, 2, new double[] { 0, 0 }, 0.1);
        network.addLayer(2, 2, new double[] { 0, 0 }, 0.1);

        Savings saver = new Savings("weights.txt");
        File weightFile = new File("weights.txt");

        if (weightFile.exists() && weightFile.length() > 0) {
            try {
                network.setWeight(saver.loadMain());
                System.out.println("Weights loaded from file.");
            } catch (Exception e) {
                System.out.println("Error loading file, starting fresh.");
            }
        } else {
            System.out.println("No weights file found. Starting training from scratch...");
        }

        int size = 10;
        int combinations = (int) Math.pow(2, size);

        double[][] inputs = new double[combinations][size];
        double[] targets = new double[combinations];

        for (int i = 0; i < combinations; i++) {
            int onesCount = 0;
            for (int j = 0; j < size; j++) {

                inputs[i][j] = (i >> (size - 1 - j)) & 1;

                if (inputs[i][j] == 1.0) {
                    onesCount++;
                }
            }

            targets[i] = (onesCount % 2 != 0) ? 1.0 : 0.0;
        }

        System.out.println("Training started...");
        for (int epoch = 0; epoch < 10000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                final int index = i;
                network.setInputs(inputs[index]);
                network.predictFuture(inputs[index], 10);
                task.doIt((progress) -> {
                    network.trainLayers(inputs[index], targets[index]);
                    return null;
                });
            }
        }
        task.waitForCompletion();

        for (int i = 0; i < inputs.length; i++) {
            network.setInputs(inputs[i]);
            double output = network.predict(targets[i]);
            if (Math.abs(targets[i] - output) < 0.15) {
                System.out.println("DID GOOD! Output: " + String.format("%.4f", output) + " | Target: " + targets[i]);
            } else {
                System.out.println("DID BAD(  Output: " + String.format("%.4f", output) + " | Target: " + targets[i]);
            }
        }

        saver.saveMain(network.getWeight());
        System.out.println("Weights saved to weights.txt");
    }
}