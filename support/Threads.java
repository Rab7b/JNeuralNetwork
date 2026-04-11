package support;
import java.lang.Thread;
import java.util.function.*;

public class Threads {
    private static final int cores = Runtime.getRuntime().availableProcessors();
    private static int usingCores = 0;

    public void doIt(Function<Double, Void> func){
        if(usingCores >= cores){
            return;
        }
        for (int i = 0; i < cores; i++) {
            new Thread(() -> {
                usingCores++;
                func.apply((double)usingCores / cores);
            }).start();
        }
    }
}
