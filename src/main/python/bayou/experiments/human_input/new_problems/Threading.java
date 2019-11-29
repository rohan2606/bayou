import java.util.*;
import java.io.*;


public class ThreadQueue(){

    public final boolean isQueued(Thread thread) {
        if (thread == null)
            throw new NullPointerException();
        for (Node p = tail; p != null; p = p.prev)
            if (p.thread == thread)
                return true;
        return false;
    }

    public final int getQueueLength() {
        int n = 0;
        for (Node p = tail; p != null; p = p.prev) {
            if (p.thread != null)
                ++n;
        }
        return n;
    }

    /**
    * get the collection of queued threads
    */
    public final Collection<Thread> getThreadsInQueue() {
        __PDB_FILL__();
    }

}
