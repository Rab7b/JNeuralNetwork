package lib.plugins;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

import java.io.File;

import lib.ai.network.*;
import lib.ai.neuron.Savings;

public class Mnist {

    private String path = "E:\\MNIST\\MNIST Dataset JPG format\\MNIST - JPG - training\\";

    public double[] parse(String path) {
        try {
            File file = new File(path);
            if (!file.exists()) {
                System.out.println("Failed");
                return null;
            }

            BufferedImage img = ImageIO.read(file);
            if (img == null) {
                System.out.println("Failed");
                return null;
            }

            if (img.getWidth() != 28 || img.getHeight() != 28) {
                System.out.println("Failed");
                return null;
            }

            double[] inputs = new double[784];
            int n = 0;

            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    int rgb = img.getRGB(x, y);

                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = rgb & 0xFF;

                    double gray = (r + g + b) / 3.0;

                    inputs[n++] = gray / 255.0;
                }
            }
            return inputs;

        } catch (Exception e) {
            System.out.println("Failed");
            return null;
        }
    }

    public void train() throws Exception {
        Savings saver = new Savings("weights.txt");
        Model network = new Model(new int[] { 784, 64, 32, 10 }, 0.001);

        int iterations = 60000;
        java.util.Random random = new java.util.Random();

        System.out.println("Starting training with random sampling...");

        for (int epoch = 0; epoch < 15; epoch++) {

            System.out.println("Epoch: " + (epoch + 1));

            for (int i = 0; i < iterations; i++) {
                int digit = random.nextInt(10);
                File folder = new File(this.path + digit);
                File[] images = folder.listFiles();

                if (images == null || images.length == 0)
                    continue;

                File randomImg = images[random.nextInt(images.length)];

                if (!randomImg.getName().toLowerCase().endsWith(".jpg"))
                    continue;

                double[] inputs = parse(randomImg.getAbsolutePath());
                if (inputs == null)
                    continue;

                double[] targets = new double[10];
                targets[digit] = 1.0;

                network.train(inputs, targets);
            }
            test();
            saver.saveMain(network.getWeightsFlat());
        }

        System.out.println("Training finished and weights saved.");
    }

    public void test() throws Exception {

        Model network = new Model(new int[] { 784, 64, 32, 10 }, 0);

        Savings loader = new Savings("weights.txt");
        double[] loadedWeights = loader.loadMain();
        network.setWeightsFlat(loadedWeights);

        System.out.println("Weights loaded. Starting test...");
        String testPath = "E:\\MNIST\\MNIST Dataset JPG format\\MNIST - JPG - testing\\";
        int totalImages = 0;
        int correct = 0;

        for (int digit = 0; digit <= 9; digit++) {
            File folder = new File(testPath + digit);
            File[] images = folder.listFiles();

            if (images != null) {
                for (File imgFile : images) {
                    if (!imgFile.getName().toLowerCase().endsWith(".jpg"))
                        continue;

                    double[] inputs = parse(imgFile.getAbsolutePath());
                    if (inputs == null)
                        continue;

                    double[] output = network.predict(inputs);
                    int prediction = getArgMax(output);

                    if (prediction == digit) {
                        correct++;
                    }
                    totalImages++;
                }
            }
        }

        double accuracy = (double) correct / totalImages * 100;
        System.out.printf("Test completed. Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, totalImages);
    }

    private int getArgMax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}