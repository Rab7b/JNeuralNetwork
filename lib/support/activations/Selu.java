package lib.support.activations;
import lib.support.*;

public class Selu implements Activation {
    private final double ALPHA = 1.67326;
    private final double SCALE = 1.0507;
    public double forward(double x) { 
        return SCALE * (x > 0 ? x : ALPHA * (Math.exp(x) - 1)); 
    }
    public double derivative(double x) {
        return SCALE * (x > 0 ? 1.0 : ALPHA * Math.exp(x));
    }
}