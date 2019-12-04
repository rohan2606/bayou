import java.sql.*;
import java.util.*;


public class SQLCore
{
    public Logger log;
    public Connection connection;
    private String host;
    private String username;
    private String password;
    private String database;
    private int port;


    public void initialize()
    {
        try
        {
            Class.forName("com.mysql.jdbc.Driver");
            connection = DriverManager.getConnection("jdbc:mysql://" + host + ":" + port + "/" + database, username, password);
        }
        catch (ClassNotFoundException e)
        {
            log.severe("ClassNotFoundException! " + e.getMessage());
        }
        catch (SQLException e)
        {
            log.severe("SQLException! " + e.getMessage());
        }
    }

    /** 
    Execute a select statement in SQL
    */
    public ResultSet select(String query)
    {
       __PDB_FILL__();
    }

}
