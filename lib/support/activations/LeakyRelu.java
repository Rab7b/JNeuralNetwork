package lib.support.activations;
import lib.support.*;

public class LeakyRelu implements Activation {
    public double forward(double x) { return x > 0 ? x : 0.01 * x; }
    public double derivative(double x) { return x > 0 ? 1.0 : 0.01; }
}