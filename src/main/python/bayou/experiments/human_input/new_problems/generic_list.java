import java.io.*;
import java.utils.*;

public class GenericList{

    public Object[] iValues;
    public int iSize;

    public void add(Object x){
        insert(iSize, x);
    }

    public Object get(int pos){
        return iValues[pos];
    }

    /**
    remove item from list
    */
    public void remove(int pos){
        __PDB_FILL__();
    }
}

