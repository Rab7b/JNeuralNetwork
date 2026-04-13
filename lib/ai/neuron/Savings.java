package lib.ai.neuron;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

public class Savings {
    private String mainFile;

    public Savings(String mainPath) {
        this.mainFile = mainPath;
    }

    public void saveMain(double[] content) throws Exception {
        Path path = Path.of(mainFile);
        try (BufferedWriter writer = Files.newBufferedWriter(path)) {
            for (double val : content) { 
                writer.write(Double.toString(val));
                writer.newLine();
            }
        }
    }

    public void save(double[] content, String path) throws Exception {
        Path path2 = Path.of(path);
        try (BufferedWriter writer = Files.newBufferedWriter(path2)) {
            for (double val : content) {
                writer.write(Double.toString(val));
                writer.newLine();
            }
        }
    }

    public double[] loadMain() throws Exception {
        try (Stream<String> lines = Files.lines(Path.of(mainFile))) {
            return lines.mapToDouble(Double::parseDouble).toArray();
        }
    }

    public double[] load(String path) throws Exception {
        try (Stream<String> lines = Files.lines(Path.of(path))) {
            return lines.mapToDouble(Double::parseDouble).toArray();
        }
    }
}