import java.util.*;
import java.io.*;

public class Stopwatch
{
    public long startTime;
    public long stopTime;

    public static final double NANOS_PER_SEC = 1000000000.0;


    public void start(){
    	startTime = System.nanoTime();
    }

    
    public void stop()
    {	stopTime = System.nanoTime();	}
   
 
    /**
    return the time recorded on the stopwatch in miiliseconds
    */
    public double time()
    {
       __PDB_FILL__(); 
    }


}

