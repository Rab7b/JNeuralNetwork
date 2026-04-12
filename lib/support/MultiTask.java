package lib.support;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.DoubleFunction;

public class MultiTask {

    private static final int CORES = Runtime.getRuntime().availableProcessors();

    private static final AtomicInteger usingCores = new AtomicInteger(0);

    private final ExecutorService executor = Executors.newFixedThreadPool(CORES);

    public void doIt(DoubleFunction<?> func) {

        executor.submit(() -> {
            int current = usingCores.incrementAndGet();

            double progress = (double) current / CORES;

            func.apply(progress);

            usingCores.decrementAndGet();
            return null;
        });
    }

    public void shutdown() {
        executor.shutdown();
    }

    public void waitForCompletion() {
        executor.shutdown(); 
        try {
            if (!executor.awaitTermination(1, TimeUnit.MINUTES)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
}