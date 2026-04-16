package lib.support.activations;
import lib.support.*;

public class Softplus implements Activation {
    public double forward(double x) { return Math.log(1.0 + Math.exp(x)); }
    public double derivative(double x) { return 1.0 / (1.0 + Math.exp(-x)); } // Это та же сигмоида!
}