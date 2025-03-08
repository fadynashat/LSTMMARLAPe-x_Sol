using Microsoft.ML.Data;

public class ModelOutput
{
    // This attribute specifies the name of the output column in the ML.NET model.
    // "Score" is the default name for the predicted value in regression models.
    [ColumnName("Score")]
    public float PredictedResponseTime { get; set; }
}
