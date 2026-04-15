package lib.support.activations;
import lib.support.*;

public class Sigmoid implements Activation {
    public double forward(double x) { return 1.0 / (1.0 + Math.exp(-x)); }
    public double derivative(double x) {
        double s = forward(x);
        return s * (1.0 - s);
    }
}