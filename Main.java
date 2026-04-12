import lib.ai.network.*;
import lib.support.*;

public class Main {
    public static void main(String[] args) {
        MultiTask mainThread = new MultiTask();
        Network network = new Network(1, 2, null, 0.01);
        double[][] inputs = {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
        };
        double[] targets = { 0, 1, 1, 1 };
        for (int epoch = 0; epoch < 10000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                final int index = i;
                network.setInputs(inputs[i]);
                mainThread.doIt(x -> {
                    network.train(targets[index]);
                    return null;
                });
            }
        }
        for (int i = 0; i < inputs.length; i++) {
            network.setInputs(inputs[i]);
            double output = network.predict(targets[i]);
            System.out.println( "Input: " + inputs[i][0] + ", " + inputs[i][1] + " Output: " + output + " Target: " + targets[i]);
        }
    }
}
