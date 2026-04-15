package lib.support.activations;
import lib.support.*;

public class Tanh implements Activation {
    public double forward(double x) { return Math.tanh(x); }
    public double derivative(double x) {
        double t = Math.tanh(x);
        return 1.0 - t * t;
    }
}