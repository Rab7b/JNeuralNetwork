package lib.support.activations;
import lib.support.*;

public class Elu implements Activation {
    private double alpha = 1.0;
    public double forward(double x) { return x > 0 ? x : alpha * (Math.exp(x) - 1); }
    public double derivative(double x) { return x > 0 ? 1.0 : forward(x) + alpha; }
}