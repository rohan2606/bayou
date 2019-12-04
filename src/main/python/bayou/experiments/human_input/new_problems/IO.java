import java.io.*;
import java.util.*;

class IO {

	public static void readFully(final InputStream fd, final byte[] dst,
			int off, int len) throws IOException {
		while (len > 0) {
			final int r = fd.read(dst, off, len);
			if (r <= 0)
				throw new EOFException(JGitText.get().shortReadOfBlock);
			off += r;
			len -= r;
		}
	}

        /**
        write bytes to outputstream
        */
	public void write(OutputStream a){
                __PDB_FILL__();
	}

}
