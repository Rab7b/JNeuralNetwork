package lib.support;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.DoubleFunction;

public class MultiTask {
    private static final int CORES = Runtime.getRuntime().availableProcessors();

    private static final AtomicInteger usingCores = new AtomicInteger(0);

    public void doIt(DoubleFunction<?> func) {

        if (usingCores.get() >= CORES) {
            return;
        }

        for (int i = 0; i < CORES; i++) {
            new Thread(() -> {

                int current = usingCores.incrementAndGet();

                double progress = (double) current / CORES;
                func.apply(progress);

                usingCores.decrementAndGet();
            }).start();
        }
    }
}
