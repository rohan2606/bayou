import java.util.*;
import java.io.*;

class Parser{

       public static String[] split(String toSplit, String separator) {
                StringTokenizer tokenizer = new StringTokenizer(toSplit, separator);
                String[] result = new String[tokenizer.countTokens()];
                int index = 0;
                while (tokenizer.hasMoreElements()) {
                        result[index] = tokenizer.nextToken().trim();
                        index++;
                }
                return result;
        }


        /**
            parse a JSON String and put it into hashtable
        */
        public static Hashtable<String,Object> JSONStringToHashtable(final String string) {
        __PDB_FILL__();
        }




}

