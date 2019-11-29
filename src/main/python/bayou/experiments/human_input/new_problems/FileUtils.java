import java.io.*;
import java.util.*;

public class FileUtils {

     public static boolean delete(File file) {
         if (!file.exists()) {
             return true;
         }
         return file.delete();
     }

     public static boolean exist(String fname) {
         String fileString = fname;
         File file = new File(fileString);
         if (file.exists()) {
            return true;
         }
            return false;
     }

     /**
     copy a file to another location
     */
     public static void copy(File from, File to){
           __PDB_FILL__(); 
     }
}
