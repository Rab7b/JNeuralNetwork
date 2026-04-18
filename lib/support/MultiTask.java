package lib.support;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.DoubleFunction;

public class MultiTask {

    private static final int CORES = Runtime.getRuntime().availableProcessors();
    private final AtomicInteger usingCores = new AtomicInteger(0);

    private final BlockingQueue<Runnable> queue = new LinkedBlockingQueue<>(CORES * 2);
    
    private final ExecutorService executor;

    public MultiTask() {
        this.executor = new ThreadPoolExecutor(
            CORES, 
            CORES, 
            0L, TimeUnit.MILLISECONDS,
            queue,
            new ThreadPoolExecutor.CallerRunsPolicy() 
        );
    }

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