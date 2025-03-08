// Enhanced LSTM Module for Workload Prediction
public class LSTM
{
    private int inputSize;  // Number of input features per time step
    private int hiddenSize; // Number of units in the hidden state
    private double[] hiddenState; // Stores the hidden state vector
    private double[] cellState;   // Stores the cell state vector
    private double[][] weights;   // Weights for input, forget, output, and cell state gates
    private double[] biases;      // Bias terms for each gate
    private double learningRate;  // Learning rate for optimization

    // Constructor to initialize LSTM parameters
    public LSTM(int inputSize, int hiddenSize, double learningRate = 0.01)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.learningRate = learningRate;

        // Initialize hidden state and cell state with zeros
        hiddenState = new double[hiddenSize];
        cellState = new double[hiddenSize];

        // Initialize weight matrices for the 4 gates: input, forget, output, and cell state
        weights = new double[4][];
        for (int i = 0; i < 4; i++)
        {
            // Each gate has weights for both input and previous hidden state
            weights[i] = new double[inputSize + hiddenSize];
            for (int j = 0; j < weights[i].Length; j++)
            {
                weights[i][j] = new Random().NextDouble() * 0.1; // Initialize with small random values
            }
        }

        // Initialize biases for the 4 gates with small random values
        biases = new double[4];
        for (int i = 0; i < 4; i++)
        {
            biases[i] = new Random().NextDouble() * 0.1;
        }
    }

    // Sigmoid activation function (used in LSTM gates)
    private double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    // Tanh activation function (used in cell state update)
    private double Tanh(double x)
    {
        return Math.Tanh(x);
    }

    // Forward pass for a single time step in LSTM
    private (double[] hiddenState, double[] cellState) ForwardPass(double[] input, double[] prevHiddenState, double[] prevCellState)
    {
        // Combine input and previous hidden state into a single vector
        var combinedInput = new double[input.Length + prevHiddenState.Length];
        input.CopyTo(combinedInput, 0);
        prevHiddenState.CopyTo(combinedInput, input.Length);

        // Compute gate activations using dot product with weights and adding biases
        var forgetGate = Sigmoid(DotProduct(weights[0], combinedInput) + biases[0]);  // Forget gate
        var inputGate = Sigmoid(DotProduct(weights[1], combinedInput) + biases[1]);   // Input gate
        var outputGate = Sigmoid(DotProduct(weights[2], combinedInput) + biases[2]);  // Output gate
        var candidateCellState = Tanh(DotProduct(weights[3], combinedInput) + biases[3]); // Candidate cell state

        // Update cell state using the forget gate and input gate
        var cellState = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            cellState[i] = forgetGate * prevCellState[i] + inputGate * candidateCellState;
        }

        // Compute hidden state using the output gate and updated cell state
        var hiddenState = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            hiddenState[i] = outputGate * Tanh(cellState[i]);
        }

        return (hiddenState, cellState);
    }

    // Predicts values for a sequence of inputs
    public double[] Predict(double[][] inputs)
    {
        var predictions = new double[inputs.Length];
        var prevHiddenState = new double[hiddenSize]; // Initialize hidden state
        var prevCellState = new double[hiddenSize];   // Initialize cell state

        // Iterate over each time step in the input sequence
        for (int t = 0; t < inputs.Length; t++)
        {
            // Perform forward pass for each time step
            var (hiddenState, cellState) = ForwardPass(inputs[t], prevHiddenState, prevCellState);

            // Simulate a prediction by summing the hidden state values
            predictions[t] = hiddenState.Sum();

            // Update previous hidden state and cell state
            prevHiddenState = hiddenState;
            prevCellState = cellState;
        }

        return predictions;
    }

    // Training function for the LSTM model
    public void Train(double[][] inputs, double[] targets, int epochs = 100, int batchSize = 32)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int batchStart = 0; batchStart < inputs.Length; batchStart += batchSize)
            {
                // Define batch size limits
                int batchEnd = Math.Min(batchStart + batchSize, inputs.Length);
                var batchInputs = inputs[batchStart..batchEnd];
                var batchTargets = targets[batchStart..batchEnd];

                // Forward pass to compute predictions
                var predictions = Predict(batchInputs);

                // Compute loss using Mean Squared Error (MSE)
                double loss = 0;
                for (int i = 0; i < predictions.Length; i++)
                {
                    loss += Math.Pow(predictions[i] - batchTargets[i], 2);
                }
                loss /= predictions.Length; // Average loss

                // Backward pass (simple gradient descent update)
                for (int i = 0; i < weights.Length; i++)
                {
                    for (int j = 0; j < weights[i].Length; j++)
                    {
                        // Adjust weights based on the loss
                        weights[i][j] -= learningRate * loss;
                    }
                }

                for (int i = 0; i < biases.Length; i++)
                {
                    // Adjust biases based on the loss
                    biases[i] -= learningRate * loss;
                }

                // Print loss for monitoring training progress
                Console.WriteLine($"Epoch {epoch + 1}, Batch {batchStart / batchSize + 1}, Loss: {loss}");
            }
        }
    }

    // Helper method to compute the dot product between two vectors
    private double DotProduct(double[] a, double[] b)
    {
        double result = 0;
        for (int i = 0; i < a.Length; i++)
        {
            result += a[i] * b[i];
        }
        return result;
    }
}
