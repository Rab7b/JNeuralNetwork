package lib.support.activations;
import lib.support.*;

public class Relu implements Activation {
    @Override
    public double forward(double x) {
        return Math.max(0, x);
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
}