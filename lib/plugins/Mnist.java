package lib.plugins;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

import java.io.File;

import lib.ai.network.*;
import lib.ai.neuron.Savings;
import lib.support.MultiTask;

public class Mnist {

    public double[] parse(String path) {
        try {
            File file = new File(path);
            if (!file.exists()) {
                System.out.println("File not found: " + path);
                return null;
            }

            BufferedImage img = ImageIO.read(file);
            if (img == null)
                return null;

            double[] inputs = new double[784];
            int n = 0;

            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    int rgb = img.getRGB(x, y);

                    int r = (rgb >> 16) & 0xFF;
                    inputs[n++] = r / 255.0;
                }
            }
            return inputs;
        } catch (Exception e) {
            return null;
        }
    }

    public void test() throws Exception {
        Savings saver = new Savings("weights.txt");
        MultiTask manager = new MultiTask();

        Model network = new Model(new int[] { 784, 64, 32, 10 }, 0.01);
        String basePath = "E:\\MNIST\\MNIST Dataset JPG format\\MNIST - JPG - training\\";

        for (int digit = 0; digit <= 9; digit++) {
            File folder = new File(basePath + digit);
            File[] images = folder.listFiles();

            if (images != null) {
                System.out.println("Training digit: " + digit + " (" + images.length + " images)");

                for (int i = 0; i < images.length; i++) {

                    if (!images[i].getName().toLowerCase().endsWith(".jpg"))
                        continue;

                    double[] inputs = parse(images[i].getAbsolutePath());

                    if (inputs == null) {
                        System.out.println("Skip null image: " + images[i].getName());
                        continue;
                    }

                    double[] targets = new double[10];
                    targets[digit] = 1.0;

                    final double[] currentInputs = inputs;
                    final double[] currentTargets = targets;

                    network.setInputs(currentInputs);

                    manager.doIt((progress) -> {
                        network.train(currentInputs, currentTargets);
                        return null;
                    });

                    if (i % 1000 == 0)
                        System.out.println("Processed " + i + " images for digit " + digit);
                }
            }
        }
        saver.saveMain(network.getWeightsFlat());
    }
}