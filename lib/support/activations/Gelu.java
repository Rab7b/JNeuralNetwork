package lib.support.activations;

import lib.support.Activation;

public class Gelu implements Activation {
    private static final double SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);
    private static final double COEFFICIENT = 0.044715;

    @Override
    public double forward(double x) {
        double xCubed = x * x * x;
        double inner = SQRT_2_OVER_PI * (x + COEFFICIENT * xCubed);
        return 0.5 * x * (1.0 + Math.tanh(inner));
    }

    @Override
    public double derivative(double x) {

        double xSq = x * x;
        double xCubed = xSq * x;
        double inner = SQRT_2_OVER_PI * (x + COEFFICIENT * xCubed);
        double tanhInner = Math.tanh(inner);

        double sech2Inner = 1.0 - (tanhInner * tanhInner);
        double dInner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFFICIENT * xSq);
        
        return 0.5 * (1.0 + tanhInner) + (0.5 * x * sech2Inner * dInner);
    }
}