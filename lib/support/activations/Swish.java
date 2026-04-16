package lib.support.activations;
import lib.support.*;

public class Swish implements Activation {
    public double forward(double x) { return x / (1.0 + Math.exp(-x)); }
    public double derivative(double x) {
        double s = 1.0 / (1.0 + Math.exp(-x));
        return s + (x * s * (1.0 - s));
    }
}