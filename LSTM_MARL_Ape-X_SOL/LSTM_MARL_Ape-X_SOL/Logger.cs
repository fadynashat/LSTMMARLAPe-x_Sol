// Logger class to handle logging
public static class Logger
{
    private static string logFile = Path.Combine(Directory.GetCurrentDirectory(), "Output", "log.txt");

    static Logger()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(logFile));
        File.WriteAllText(logFile, "Log Started\n"); // Initialize log file
    }

    public static void Log(string message)
    {
        string logEntry = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} - {message}";
        Console.WriteLine(logEntry);
        File.AppendAllText(logFile, logEntry + Environment.NewLine);
    }
}