import java.io.*;
import java.util.*;

class PeekingIterator {
    private Integer next = null;
    private Iterator<T> iter;

    public T peek() {
        return next;
    }

    public boolean hasNext() {
        return next != null;
    }

    /**
    advance iterator to next element if it exists
    */    
    public T next() {
        __PDB_FILL__();
    }

}

