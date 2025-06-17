using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AIMouseTrainer
{
    public partial class PredictorForm : Form
    {
        const bool TRAINING_MODE = true; // eğitim bittiğinde test etmek için kapat

        const int WIDTH = 800, HEIGHT = 600;
        const int FPS = 200;
        const int TARGET_RADIUS = 15;

        const int MOUSE_HISTORY_SEQUENCE_LENGTH = 15;
        const int FEATURES_PER_TIMESTEP = 9;
        const int LSTM_HIDDEN_SIZE = 64;
        const int HISTORY_POINTS_NEEDED_FOR_FEATURES = 4;
        const int MAX_POINTS = MOUSE_HISTORY_SEQUENCE_LENGTH + HISTORY_POINTS_NEEDED_FOR_FEATURES + 50;

        const double MOUSE_STEP_SIZE = 4.0;
        const double MAX_SPEED_INPUT_NORMALIZATION = MOUSE_STEP_SIZE * 3.0;
        const double MIN_SPEED_MULTIPLIER = 0.2;
        const double MAX_SPEED_MULTIPLIER = 3.0;
        const double MAX_ACCELERATION_NORMALIZATION = 2.0 * MOUSE_STEP_SIZE;
        const double ANGULAR_CHANGE_NORMALIZATION_DIVISOR = Math.PI;

        const int BATCH_SIZE = 32;
        const double GRADIENT_CLIP_THRESHOLD = 1.0;

        public static class MatrixMath
        {
            private static readonly int VectorWidth = Vector<double>.Count;

            public static double[] Add(double[] a, double[] b)
            {
                var result = new double[a.Length];
                int i = 0;
                for (; i <= a.Length - VectorWidth; i += VectorWidth)
                {
                    var va = new Vector<double>(a, i);
                    var vb = new Vector<double>(b, i);
                    (va + vb).CopyTo(result, i);
                }
                for (; i < a.Length; i++) result[i] = a[i] + b[i];
                return result;
            }

            public static double[] Multiply(double[] a, double[] b)
            {
                var result = new double[a.Length];
                int i = 0;
                for (; i <= a.Length - VectorWidth; i += VectorWidth)
                {
                    var va = new Vector<double>(a, i);
                    var vb = new Vector<double>(b, i);
                    (va * vb).CopyTo(result, i);
                }
                for (; i < a.Length; i++) result[i] = a[i] * b[i];
                return result;
            }

            public static double[] MatVecMul(double[,] m, double[] v, double[] bias)
            {
                int hiddenSize = m.GetLength(1);
                int inputSize = m.GetLength(0);
                var result = new double[hiddenSize];
                Array.Copy(bias, result, bias.Length);

                for (int i = 0; i < hiddenSize; i++)
                {
                    double sum = 0;
                    int j = 0;
                    for (; j <= inputSize - VectorWidth; j += VectorWidth)
                    {
                        var v_vec = new Vector<double>(v, j);
                        var m_col_slice = new double[VectorWidth];
                        for (int k = 0; k < VectorWidth; k++) m_col_slice[k] = m[j + k, i];
                        var m_vec = new Vector<double>(m_col_slice);
                        sum += Vector.Dot(v_vec, m_vec);
                    }
                    for (; j < inputSize; j++) sum += m[j, i] * v[j];
                    result[i] += sum;
                }
                return result;
            }

            public static double[] MatMulTranspose(double[,] m, double[] v)
            {
                int rows = m.GetLength(0);
                int cols = m.GetLength(1);
                var result = new double[rows];
                for (int i = 0; i < rows; i++)
                {
                    double sum = 0;
                    int j = 0;
                    for (; j <= cols - VectorWidth; j += VectorWidth)
                    {
                        var m_row_slice = new double[VectorWidth];
                        for (int k = 0; k < VectorWidth; k++) m_row_slice[k] = m[i, j + k];

                        var m_vec = new Vector<double>(m_row_slice);
                        var v_vec = new Vector<double>(v, j);
                        sum += Vector.Dot(m_vec, v_vec);
                    }
                    for (; j < cols; j++) sum += m[i, j] * v[j];
                    result[i] = sum;
                }
                return result;
            }
        }

        public class GradientContainer
        {
            public Dictionary<string, object> Gradients { get; } = new Dictionary<string, object>();
            public object PropagatedError { get; set; }
        }

        public abstract class Layer
        {
            public string Name { get; protected set; }
            public abstract object Forward(object input);
            public abstract GradientContainer ComputeGradients(object error);
            public abstract void ApplyGradients(GradientContainer gradientContainer, AdamOptimizer optimizer, double clipScale);
            public abstract object Save();
            public abstract void Load(JsonElement data);
        }

        public class LSTMLayer : Layer
        {
            private readonly int _inputSize, _hiddenSize;
            private static readonly Random Rng = new Random();

            private double[,] Wf, Wi, Wg, Wo, Uf, Ui, Ug, Uo;
            private double[] Bf, Bi, Bg, Bo;

            private List<double[]> _lastInputSequence, _lastHiddenStates, _lastCellStates;
            private List<double[]> _lastForgetGates, _lastInputGates, _lastOutputGates, _lastCellCandidates;

            private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
            private static double Tanh(double x) => Math.Tanh(x);
            private static double SigmoidDerivative(double x) => x * (1 - x);
            private static double TanhDerivative(double x) => 1 - (x * x);

            public LSTMLayer(int inputSize, int hiddenSize, string name)
            {
                Name = name;
                _inputSize = inputSize;
                _hiddenSize = hiddenSize;
                double scale = Math.Sqrt(6.0 / (inputSize + hiddenSize));

                Wf = InitializeMatrix(inputSize, hiddenSize, scale); Wi = InitializeMatrix(inputSize, hiddenSize, scale);
                Wg = InitializeMatrix(inputSize, hiddenSize, scale); Wo = InitializeMatrix(inputSize, hiddenSize, scale);
                Uf = InitializeMatrix(hiddenSize, hiddenSize, scale); Ui = InitializeMatrix(hiddenSize, hiddenSize, scale);
                Ug = InitializeMatrix(hiddenSize, hiddenSize, scale); Uo = InitializeMatrix(hiddenSize, hiddenSize, scale);

                Bf = new double[hiddenSize]; Bi = new double[hiddenSize];
                Bg = new double[hiddenSize]; Bo = new double[hiddenSize];
            }

            private double[,] InitializeMatrix(int r, int c, double s) { var m = new double[r, c]; for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) m[i, j] = (Rng.NextDouble() * 2 - 1) * s; return m; }

            public override object Forward(object input)
            {
                var inputSequence = (List<double[]>)input;
                _lastInputSequence = inputSequence;
                var h = new double[_hiddenSize]; var c = new double[_hiddenSize];

                _lastHiddenStates = new List<double[]> { new double[_hiddenSize] };
                _lastCellStates = new List<double[]> { new double[_hiddenSize] };
                _lastForgetGates = new List<double[]>(); _lastInputGates = new List<double[]>();
                _lastOutputGates = new List<double[]>(); _lastCellCandidates = new List<double[]>();

                for (int t = 0; t < inputSequence.Count; t++)
                {
                    var x_t = inputSequence[t]; var h_prev = h; var c_prev = c;

                    var f_t = MatrixMath.Add(MatrixMath.MatVecMul(Wf, x_t, Bf), MatrixMath.MatVecMul(Uf, h_prev, new double[_hiddenSize])).Select(Sigmoid).ToArray();
                    var i_t = MatrixMath.Add(MatrixMath.MatVecMul(Wi, x_t, Bi), MatrixMath.MatVecMul(Ui, h_prev, new double[_hiddenSize])).Select(Sigmoid).ToArray();
                    var g_t = MatrixMath.Add(MatrixMath.MatVecMul(Wg, x_t, Bg), MatrixMath.MatVecMul(Ug, h_prev, new double[_hiddenSize])).Select(Tanh).ToArray();
                    var o_t = MatrixMath.Add(MatrixMath.MatVecMul(Wo, x_t, Bo), MatrixMath.MatVecMul(Uo, h_prev, new double[_hiddenSize])).Select(Sigmoid).ToArray();

                    c = MatrixMath.Add(MatrixMath.Multiply(f_t, c_prev), MatrixMath.Multiply(i_t, g_t));
                    h = MatrixMath.Multiply(o_t, c.Select(Tanh).ToArray());

                    _lastHiddenStates.Add(h); _lastCellStates.Add(c); _lastForgetGates.Add(f_t);
                    _lastInputGates.Add(i_t); _lastOutputGates.Add(o_t); _lastCellCandidates.Add(g_t);
                }
                return h;
            }

            public override GradientContainer ComputeGradients(object error)
            {
                var d_h = (double[])error; var d_c = new double[_hiddenSize];
                var dWf = new double[_inputSize, _hiddenSize]; var dWi = new double[_inputSize, _hiddenSize];
                var dWg = new double[_inputSize, _hiddenSize]; var dWo = new double[_inputSize, _hiddenSize];
                var dUf = new double[_hiddenSize, _hiddenSize]; var dUi = new double[_hiddenSize, _hiddenSize];
                var dUg = new double[_hiddenSize, _hiddenSize]; var dUo = new double[_hiddenSize, _hiddenSize];
                var dBf = new double[_hiddenSize]; var dBi = new double[_hiddenSize];
                var dBg = new double[_hiddenSize]; var dBo = new double[_hiddenSize];

                for (int t = _lastInputSequence.Count - 1; t >= 0; t--)
                {
                    var x_t = _lastInputSequence[t]; var h_prev = _lastHiddenStates[t]; var c_prev = _lastCellStates[t]; var c_t = _lastCellStates[t + 1];
                    var f_t = _lastForgetGates[t]; var i_t = _lastInputGates[t]; var o_t = _lastOutputGates[t]; var g_t = _lastCellCandidates[t];
                    var tanh_c_t = c_t.Select(Tanh).ToArray();

                    var d_o_t = MatrixMath.Multiply(d_h, tanh_c_t); d_o_t = MatrixMath.Multiply(d_o_t, o_t.Select(SigmoidDerivative).ToArray());
                    d_c = MatrixMath.Add(d_c, MatrixMath.Multiply(MatrixMath.Multiply(d_h, o_t), tanh_c_t.Select(TanhDerivative).ToArray()));
                    var d_g_t = MatrixMath.Multiply(d_c, i_t); d_g_t = MatrixMath.Multiply(d_g_t, g_t.Select(TanhDerivative).ToArray());
                    var d_i_t = MatrixMath.Multiply(d_c, g_t); d_i_t = MatrixMath.Multiply(d_i_t, i_t.Select(SigmoidDerivative).ToArray());
                    var d_f_t = MatrixMath.Multiply(d_c, c_prev); d_f_t = MatrixMath.Multiply(d_f_t, f_t.Select(SigmoidDerivative).ToArray());

                    AccumulateGradients(dWf, dUf, dBf, d_f_t, x_t, h_prev); AccumulateGradients(dWi, dUi, dBi, d_i_t, x_t, h_prev);
                    AccumulateGradients(dWg, dUg, dBg, d_g_t, x_t, h_prev); AccumulateGradients(dWo, dUo, dBo, d_o_t, x_t, h_prev);

                    d_c = MatrixMath.Multiply(d_c, f_t);
                    var d_h_prev = new double[_hiddenSize];
                    d_h_prev = MatrixMath.Add(d_h_prev, MatrixMath.MatMulTranspose(Uf, d_f_t)); d_h_prev = MatrixMath.Add(d_h_prev, MatrixMath.MatMulTranspose(Ui, d_i_t));
                    d_h_prev = MatrixMath.Add(d_h_prev, MatrixMath.MatMulTranspose(Ug, d_g_t)); d_h_prev = MatrixMath.Add(d_h_prev, MatrixMath.MatMulTranspose(Uo, d_o_t));
                    d_h = d_h_prev;
                }

                var container = new GradientContainer { PropagatedError = d_h };
                container.Gradients.Add("Wf", dWf); container.Gradients.Add("Wi", dWi); container.Gradients.Add("Wg", dWg); container.Gradients.Add("Wo", dWo);
                container.Gradients.Add("Uf", dUf); container.Gradients.Add("Ui", dUi); container.Gradients.Add("Ug", dUg); container.Gradients.Add("Uo", dUo);
                container.Gradients.Add("Bf", dBf); container.Gradients.Add("Bi", dBi); container.Gradients.Add("Bg", dBg); container.Gradients.Add("Bo", dBo);
                return container;
            }

            public override void ApplyGradients(GradientContainer gradientContainer, AdamOptimizer optimizer, double clipScale)
            {
                var grads = gradientContainer.Gradients;
                optimizer.Update(Wf, (double[,])grads["Wf"], $"{Name}_Wf", clipScale); optimizer.Update(Wi, (double[,])grads["Wi"], $"{Name}_Wi", clipScale);
                optimizer.Update(Wg, (double[,])grads["Wg"], $"{Name}_Wg", clipScale); optimizer.Update(Wo, (double[,])grads["Wo"], $"{Name}_Wo", clipScale);
                optimizer.Update(Uf, (double[,])grads["Uf"], $"{Name}_Uf", clipScale); optimizer.Update(Ui, (double[,])grads["Ui"], $"{Name}_Ui", clipScale);
                optimizer.Update(Ug, (double[,])grads["Ug"], $"{Name}_Ug", clipScale); optimizer.Update(Uo, (double[,])grads["Uo"], $"{Name}_Uo", clipScale);
                optimizer.Update(Bf, (double[])grads["Bf"], $"{Name}_Bf", clipScale); optimizer.Update(Bi, (double[])grads["Bi"], $"{Name}_Bi", clipScale);
                optimizer.Update(Bg, (double[])grads["Bg"], $"{Name}_Bg", clipScale); optimizer.Update(Bo, (double[])grads["Bo"], $"{Name}_Bo", clipScale);
            }

            private void AccumulateGradients(double[,] dW, double[,] dU, double[] dB, double[] delta, double[] x, double[] h_prev)
            {
                for (int i = 0; i < _hiddenSize; i++)
                {
                    for (int j = 0; j < _inputSize; j++) dW[j, i] += delta[i] * x[j];
                    for (int j = 0; j < _hiddenSize; j++) dU[j, i] += delta[i] * h_prev[j];
                    dB[i] += delta[i];
                }
            }

            public override object Save() => new { type = "LSTMLayer", _inputSize, _hiddenSize, Wf = JaggedFrom2D(Wf), Wi = JaggedFrom2D(Wi), Wg = JaggedFrom2D(Wg), Wo = JaggedFrom2D(Wo), Uf = JaggedFrom2D(Uf), Ui = JaggedFrom2D(Ui), Ug = JaggedFrom2D(Ug), Uo = JaggedFrom2D(Uo), Bf, Bi, Bg, Bo };
            public override void Load(JsonElement data) { LoadMatrix(Wf, data.GetProperty("Wf")); LoadMatrix(Wi, data.GetProperty("Wi")); LoadMatrix(Wg, data.GetProperty("Wg")); LoadMatrix(Wo, data.GetProperty("Wo")); LoadMatrix(Uf, data.GetProperty("Uf")); LoadMatrix(Ui, data.GetProperty("Ui")); LoadMatrix(Ug, data.GetProperty("Ug")); LoadMatrix(Uo, data.GetProperty("Uo")); LoadVector(Bf, data.GetProperty("Bf")); LoadVector(Bi, data.GetProperty("Bi")); LoadVector(Bg, data.GetProperty("Bg")); LoadVector(Bo, data.GetProperty("Bo")); }
            private double[][] JaggedFrom2D(double[,] a) { var j = new double[a.GetLength(0)][]; for (int i = 0; i < a.GetLength(0); i++) { j[i] = new double[a.GetLength(1)]; for (int k = 0; k < a.GetLength(1); k++) j[i][k] = a[i, k]; } return j; }
            private void LoadMatrix(double[,] t, JsonElement s) { int i = 0; foreach (var r in s.EnumerateArray()) { int j = 0; foreach (var v in r.EnumerateArray()) { if (i < t.GetLength(0) && j < t.GetLength(1)) t[i, j] = v.GetDouble(); j++; } i++; } }
            private void LoadVector(double[] t, JsonElement s) { int i = 0; foreach (var v in s.EnumerateArray()) { if (i < t.Length) t[i++] = v.GetDouble(); } }
        }

        public class DenseLayer : Layer
        {
            readonly int inSize, outSize; string activation;
            readonly double[,] W; readonly double[] B;
            double[] lastInput, lastOutput;
            static readonly Random rng = new Random();


            public DenseLayer(int i, int o, string a, string name)
            {
                Name = name;
                inSize = i; outSize = o; activation = a; W = new double[i, o]; B = new double[o];
                double s = Math.Sqrt(6.0 / (i + o));
                for (int r = 0; r < i; r++) for (int c = 0; c < o; c++) W[r, c] = (rng.NextDouble() * 2 - 1) * s;
            }

            public override object Forward(object input)
            {
                lastInput = (double[])input;
                double[] z = MatrixMath.MatVecMul(W, lastInput, B);
                if (activation == "relu") for (int j = 0; j < outSize; j++) z[j] = Math.Max(0, z[j]);
                else if (activation == "tanh") for (int j = 0; j < outSize; j++) z[j] = Math.Tanh(z[j]);
                lastOutput = z;
                return z;
            }

            public override GradientContainer ComputeGradients(object error)
            {
                var errorArray = (double[])error;
                var gradAct = new double[outSize];
                for (int j = 0; j < outSize; j++)
                    gradAct[j] = activation == "relu" ? (lastOutput[j] > 0 ? errorArray[j] : 0.0) : errorArray[j] * (1 - lastOutput[j] * lastOutput[j]);

                var dW = new double[inSize, outSize];
                var dB = new double[outSize];
                for (int i = 0; i < inSize; i++) for (int j = 0; j < outSize; j++) dW[i, j] = lastInput[i] * gradAct[j];
                Array.Copy(gradAct, dB, outSize);

                var errorPrev = MatrixMath.MatMulTranspose(W, gradAct);

                var container = new GradientContainer { PropagatedError = errorPrev };
                container.Gradients.Add("W", dW);
                container.Gradients.Add("B", dB);
                return container;
            }

            public override void ApplyGradients(GradientContainer gradientContainer, AdamOptimizer optimizer, double clipScale)
            {
                optimizer.Update(W, (double[,])gradientContainer.Gradients["W"], $"{Name}_W", clipScale);
                optimizer.Update(B, (double[])gradientContainer.Gradients["B"], $"{Name}_B", clipScale);
            }

            public override object Save() { var wJ = new double[inSize][]; for (int i = 0; i < inSize; i++) { wJ[i] = new double[outSize]; for (int j = 0; j < outSize; j++) wJ[i][j] = W[i, j]; } return new { type = "DenseLayer", inSize, outSize, activation, W = wJ, B }; }
            public override void Load(JsonElement d) { activation = d.GetProperty("activation").GetString(); var rE = d.GetProperty("W").EnumerateArray(); int i = 0; foreach (var r in rE) { int j = 0; foreach (var v in r.EnumerateArray()) { if (i < inSize && j < outSize) W[i, j] = v.GetDouble(); j++; } i++; } var bA = d.GetProperty("B").EnumerateArray().ToArray(); for (int j = 0; j < outSize && j < bA.Length; j++) B[j] = bA[j].GetDouble(); }
        }

        public class AdamOptimizer
        {
            private readonly double _lr, _beta1, _beta2, _epsilon;
            private int _timestep = 0;

            private readonly Dictionary<string, (object m, object v)> _cache = new Dictionary<string, (object, object)>();

            public AdamOptimizer(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
            {
                _lr = lr; _beta1 = beta1; _beta2 = beta2; _epsilon = epsilon;
            }

            public void IncrementTimestep() => _timestep++;

            private void EnsureCache(string key, Array param)
            {
                if (_cache.ContainsKey(key)) return;
                if (param is double[,] p2d) _cache[key] = (new double[p2d.GetLength(0), p2d.GetLength(1)], new double[p2d.GetLength(0), p2d.GetLength(1)]);
                else if (param is double[] p1d) _cache[key] = (new double[p1d.Length], new double[p1d.Length]);
            }

            public void Update(double[,] param, double[,] grad, string key, double clipScale)
            {
                EnsureCache(key, param);
                var (m, v) = ((double[,], double[,]))_cache[key];

                double m_corr_factor = 1.0 / (1.0 - Math.Pow(_beta1, _timestep));
                double v_corr_factor = 1.0 / (1.0 - Math.Pow(_beta2, _timestep));

                for (int i = 0; i < param.GetLength(0); i++)
                    for (int j = 0; j < param.GetLength(1); j++)
                    {
                        double g = grad[i, j] * clipScale;
                        m[i, j] = _beta1 * m[i, j] + (1 - _beta1) * g;
                        v[i, j] = _beta2 * v[i, j] + (1 - _beta2) * g * g;
                        double m_hat = m[i, j] * m_corr_factor;
                        double v_hat = v[i, j] * v_corr_factor;
                        param[i, j] -= _lr * m_hat / (Math.Sqrt(v_hat) + _epsilon);
                    }
            }
            public void Update(double[] param, double[] grad, string key, double clipScale)
            {
                EnsureCache(key, param);
                var (m, v) = ((double[], double[]))_cache[key];

                double m_corr_factor = 1.0 / (1.0 - Math.Pow(_beta1, _timestep));
                double v_corr_factor = 1.0 / (1.0 - Math.Pow(_beta2, _timestep));

                for (int i = 0; i < param.Length; i++)
                {
                    double g = grad[i] * clipScale;
                    m[i] = _beta1 * m[i] + (1 - _beta1) * g;
                    v[i] = _beta2 * v[i] + (1 - _beta2) * g * g;
                    double m_hat = m[i] * m_corr_factor;
                    double v_hat = v[i] * v_corr_factor;
                    param[i] -= _lr * m_hat / (Math.Sqrt(v_hat) + _epsilon);
                }
            }
        }

        class NeuralNetwork
        {
            readonly List<Layer> _layers = new List<Layer>();
            readonly AdamOptimizer _optimizer = new AdamOptimizer();

            public NeuralNetwork()
            {
                _layers.Add(new LSTMLayer(FEATURES_PER_TIMESTEP, LSTM_HIDDEN_SIZE, "lstm1"));
                _layers.Add(new DenseLayer(LSTM_HIDDEN_SIZE, 64, "relu", "dense1"));
                _layers.Add(new DenseLayer(64, 3, "tanh", "dense2"));
            }

            public double[] Forward(List<double[]> x_sequence)
            {
                object current_output = x_sequence;
                foreach (var layer in _layers)
                    current_output = layer.Forward(current_output);
                return (double[])current_output;
            }

            public void Train(List<(List<double[]> X, double[] Y)> trainingData, int epochs, Action<int, double> cb = null)
            {
                for (int ep = 0; ep < epochs; ep++)
                {
                    double epochLoss = 0;
                    var shuffledData = trainingData.OrderBy(x => Guid.NewGuid()).ToList();
                    int batchCount = 0;

                    for (int i = 0; i < shuffledData.Count; i += BATCH_SIZE)
                    {
                        var batch = shuffledData.Skip(i).Take(BATCH_SIZE).ToList();
                        if (batch.Count == 0) continue;
                        batchCount++;

                        var batchGradients = new List<List<GradientContainer>>();
                        double batchLoss = 0;

                        foreach (var sample in batch)
                        {
                            var sampleGradients = new List<GradientContainer>();
                            var output = Forward(sample.X);
                            var error = new double[output.Length];
                            double sampleLoss = 0;
                            for (int k = 0; k < error.Length; k++)
                            {
                                error[k] = output[k] - sample.Y[k];
                                sampleLoss += error[k] * error[k];
                            }
                            batchLoss += sampleLoss;

                            object back_error = error;
                            for (int l = _layers.Count - 1; l >= 0; l--)
                            {
                                var gradContainer = _layers[l].ComputeGradients(back_error);
                                sampleGradients.Insert(0, gradContainer);
                                back_error = gradContainer.PropagatedError;
                            }
                            batchGradients.Add(sampleGradients);
                        }
                        epochLoss += batchLoss / batch.Count;

                        var accumulatedGradients = new List<GradientContainer>();
                        for (int l = 0; l < _layers.Count; l++)
                        {
                            var layerGrads = new GradientContainer();
                            foreach (var gradName in batchGradients[0][l].Gradients.Keys)
                            {
                                if (batchGradients[0][l].Gradients[gradName] is double[,] template2d)
                                {
                                    var summedGrad = new double[template2d.GetLength(0), template2d.GetLength(1)];
                                    foreach (var sampleGrads in batchGradients)
                                    {
                                        var currentGrad = (double[,])sampleGrads[l].Gradients[gradName];
                                        for (int r = 0; r < template2d.GetLength(0); r++) for (int c = 0; c < template2d.GetLength(1); c++) summedGrad[r, c] += currentGrad[r, c];
                                    }
                                    for (int r = 0; r < template2d.GetLength(0); r++) for (int c = 0; c < template2d.GetLength(1); c++) summedGrad[r, c] /= batch.Count;
                                    layerGrads.Gradients.Add(gradName, summedGrad);
                                }
                                else if (batchGradients[0][l].Gradients[gradName] is double[] template1d)
                                {
                                    var summedGrad = new double[template1d.Length];
                                    foreach (var sampleGrads in batchGradients)
                                    {
                                        var currentGrad = (double[])sampleGrads[l].Gradients[gradName];
                                        for (int k = 0; k < template1d.Length; k++) summedGrad[k] += currentGrad[k];
                                    }
                                    for (int k = 0; k < template1d.Length; k++) summedGrad[k] /= batch.Count;
                                    layerGrads.Gradients.Add(gradName, summedGrad);
                                }
                            }
                            accumulatedGradients.Add(layerGrads);
                        }

                        double totalNorm = 0;
                        foreach (var layerGrads in accumulatedGradients)
                            foreach (var grad in layerGrads.Gradients.Values)
                            {
                                if (grad is double[,] g2d) foreach (var val in g2d) totalNorm += val * val;
                                else if (grad is double[] g1d) foreach (var val in g1d) totalNorm += val * val;
                            }
                        totalNorm = Math.Sqrt(totalNorm);
                        double clipScale = (totalNorm > GRADIENT_CLIP_THRESHOLD) ? GRADIENT_CLIP_THRESHOLD / totalNorm : 1.0;

                        _optimizer.IncrementTimestep();
                        for (int l = 0; l < _layers.Count; l++)
                        {
                            _layers[l].ApplyGradients(accumulatedGradients[l], _optimizer, clipScale);
                        }
                    }
                    cb?.Invoke(ep, epochLoss / batchCount);
                }
            }

            public void Save(string path)
            {
                var json = new { layers = _layers.Select(l => l.Save()).ToArray() };
                File.WriteAllText(path, JsonSerializer.Serialize(json, new JsonSerializerOptions { WriteIndented = true }));
            }
            public void Load(string path)
            {
                if (!File.Exists(path))
                {
                    Console.WriteLine($"Model dosyası bulunamadı: {path}. Varsayılan ağ yapısı kullanılacak.");
                    return;
                }
                try
                {
                    var doc = JsonDocument.Parse(File.ReadAllText(path));
                    var arr = doc.RootElement.GetProperty("layers").EnumerateArray().ToList();
                    if (arr.Count != _layers.Count) throw new Exception("Kaydedilmiş modeldeki katman sayısı mevcut modelle uyuşmuyor.");

                    for (int i = 0; i < _layers.Count; i++)
                    {
                        var layerData = arr[i];
                        var type = layerData.GetProperty("type").GetString();
                        if (_layers[i].GetType().Name != type)
                        {
                            throw new NotSupportedException($"Katman {i} tipi uyuşmuyor: Beklenen {_layers[i].GetType().Name}, bulunan {type}");
                        }
                        _layers[i].Load(layerData);
                    }
                    Console.WriteLine($"Model başarıyla yüklendi: {path}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Model yüklenirken hata oluştu ({path}): {ex.Message}. Değişiklik yapılmadı.");
                }
            }
        }

        readonly NeuralNetwork net = new NeuralNetwork();
        readonly List<(List<double[]> feats, double[] tgt)> trainingData = new List<(List<double[]>, double[])>();
        readonly Queue<PointF> mouseTrail = new Queue<PointF>(MAX_POINTS);

        PointF targetPos;
        bool autoMove = false;
        readonly Random rng = new Random();
        readonly Stopwatch fpsWatch = Stopwatch.StartNew();
        int frameCounter = 0;
        bool trainingInProgress = false;

        PointF? _last_ai_pos = null;
        Queue<PointF> _ai_pos_history_for_path_drawing = new Queue<PointF>(MAX_POINTS);
        private ProgressBar trainingProgressBar;

        public PredictorForm()
        {
            ClientSize = new Size(WIDTH, HEIGHT);
            Text = "Fare Tahmincisi (Gelişmiş LSTM) – .NET";
            DoubleBuffered = true;
            KeyPreview = true;
            targetPos = RandomTarget();

            trainingProgressBar = new ProgressBar
            {
                Name = "trainingProgressBar",
                Location = new Point(0, HEIGHT - 20),
                Width = WIDTH,
                Height = 20,
                Minimum = 0,
                Maximum = 100,
                Value = 0,
                Visible = false
            };
            this.Controls.Add(trainingProgressBar);

            PointF initialMousePos = PointToClient(Cursor.Position);
            for (int i = 0; i < HISTORY_POINTS_NEEDED_FOR_FEATURES; ++i) mouseTrail.Enqueue(initialMousePos);

            var timer = new Timer { Interval = Math.Max(1, 1000 / FPS) };
            timer.Tick += (_, __) => { UpdateLogic(); Invalidate(); };
            timer.Start();

            Paint += OnPaintCanvas;
            KeyDown += OnKeyDown;
            Load += PredictorForm_Load;
        }
        private void PredictorForm_Load(object sender, EventArgs e)
        {
            string model_path = "mouse_predictor_lstm.json";
            if (!TRAINING_MODE && File.Exists(model_path))
            {
                Console.WriteLine($"Test modu, {model_path} yükleniyor...");
                net.Load(model_path);
            }
            else if (TRAINING_MODE)
            {
                Console.WriteLine("Eğitim modu, yeni model oluşturulacak veya üzerine yazılacak.");
            }
            else
            {
                Console.WriteLine($"Test modu, ancak {model_path} bulunamadı. Varsayılan ağ kullanılacak.");
            }

            ShowMessage(TRAINING_MODE
                ? "Eğitim modundayız (LSTM). Fareyi hedeflere sürükle! (S: Eğit ve Kaydet)"
                : "Test modundayız (LSTM). V ile otomatik hareketi aç/kapat.");
        }
        PointF RandomTarget() => new PointF(rng.Next(TARGET_RADIUS, WIDTH - TARGET_RADIUS), rng.Next(TARGET_RADIUS, HEIGHT - TARGET_RADIUS));
        void ShowMessage(string msg) => MessageBox.Show(this, msg, Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
        double Distance(PointF a, PointF b) { double dx = a.X - b.X, dy = a.Y - b.Y; return Math.Sqrt(dx * dx + dy * dy); }

        double[] GetFeaturesForOneTimeStep(PointF pk, PointF pk1, PointF pk2, PointF pk3, PointF currentTargetPos)
        {
            var features = new double[FEATURES_PER_TIMESTEP];

            features[0] = (currentTargetPos.X - pk.X) / WIDTH;
            features[1] = (currentTargetPos.Y - pk.Y) / HEIGHT;

            double[] move_vec_k = { pk.X - pk1.X, pk.Y - pk1.Y };
            double speed_k = Math.Sqrt(move_vec_k[0] * move_vec_k[0] + move_vec_k[1] * move_vec_k[1]);
            features[2] = (speed_k > 0.001) ? move_vec_k[0] / speed_k : 0;
            features[3] = (speed_k > 0.001) ? move_vec_k[1] / speed_k : 0;
            features[4] = Math.Min(1.0, speed_k / MAX_SPEED_INPUT_NORMALIZATION);

            double[] move_vec_k1 = { pk1.X - pk2.X, pk1.Y - pk2.Y };
            double speed_k1 = Math.Sqrt(move_vec_k1[0] * move_vec_k1[0] + move_vec_k1[1] * move_vec_k1[1]);
            features[5] = Math.Min(1.0, speed_k1 / MAX_SPEED_INPUT_NORMALIZATION);

            double acceleration_raw = speed_k - speed_k1;
            features[6] = Math.Max(-1.0, Math.Min(1.0, acceleration_raw / MAX_ACCELERATION_NORMALIZATION));

            double angular_change_k_k1 = 0;
            if (speed_k > 0.001 && speed_k1 > 0.001)
            {
                double dot = (move_vec_k[0] * move_vec_k1[0] + move_vec_k[1] * move_vec_k1[1]) / (speed_k * speed_k1);
                double angle = Math.Acos(Math.Max(-1.0, Math.Min(1.0, dot)));
                double cross_z = move_vec_k[0] * move_vec_k1[1] - move_vec_k[1] * move_vec_k1[0];
                angular_change_k_k1 = (cross_z < 0) ? -angle : angle;
            }
            features[7] = Math.Max(-1.0, Math.Min(1.0, angular_change_k_k1 / ANGULAR_CHANGE_NORMALIZATION_DIVISOR));

            double[] move_vec_k2 = { pk2.X - pk3.X, pk2.Y - pk3.Y };
            double speed_k2 = Math.Sqrt(move_vec_k2[0] * move_vec_k2[0] + move_vec_k2[1] * move_vec_k2[1]);
            double angular_change_k1_k2 = 0;
            if (speed_k1 > 0.001 && speed_k2 > 0.001)
            {
                double dot = (move_vec_k1[0] * move_vec_k2[0] + move_vec_k1[1] * move_vec_k2[1]) / (speed_k1 * speed_k2);
                double angle = Math.Acos(Math.Max(-1.0, Math.Min(1.0, dot)));
                double cross_z = move_vec_k1[0] * move_vec_k2[1] - move_vec_k1[1] * move_vec_k2[0];
                angular_change_k1_k2 = (cross_z < 0) ? -angle : angle;
            }
            features[8] = Math.Max(-1.0, Math.Min(1.0, angular_change_k1_k2 / ANGULAR_CHANGE_NORMALIZATION_DIVISOR));

            return features;
        }

        PointF GetPointFromTrailSafely(IReadOnlyList<PointF> trail, int index, PointF defaultIfEmpty)
        {
            if (trail == null || trail.Count == 0) return defaultIfEmpty;
            if (index < 0) return trail[0];
            if (index >= trail.Count) return trail[trail.Count - 1];
            return trail[index];
        }

        List<double[]> PrepareFeatureSequence(IReadOnlyList<PointF> fullTrail, PointF currentTargetPos)
        {
            var sequence = new List<double[]>();
            PointF default_padding_point = (fullTrail != null && fullTrail.Count > 0) ? fullTrail[0] : PointF.Empty;

            for (int i = 0; i < MOUSE_HISTORY_SEQUENCE_LENGTH; i++)
            {
                PointF pk = GetPointFromTrailSafely(fullTrail, fullTrail.Count - 1 - i, default_padding_point);
                PointF pk1 = GetPointFromTrailSafely(fullTrail, fullTrail.Count - 1 - i - 1, default_padding_point);
                PointF pk2 = GetPointFromTrailSafely(fullTrail, fullTrail.Count - 1 - i - 2, default_padding_point);
                PointF pk3 = GetPointFromTrailSafely(fullTrail, fullTrail.Count - 1 - i - 3, default_padding_point);

                sequence.Insert(0, GetFeaturesForOneTimeStep(pk, pk1, pk2, pk3, currentTargetPos));
            }
            return sequence;
        }
        void UpdateLogic()
        {
            var os_mouse_pos = PointToClient(Cursor.Position);

            if (!trainingInProgress) mouseTrail.Enqueue(os_mouse_pos);
            while (mouseTrail.Count > MAX_POINTS) mouseTrail.Dequeue();

            if (autoMove && _last_ai_pos.HasValue)
            {
                _ai_pos_history_for_path_drawing.Enqueue(_last_ai_pos.Value);
                while (_ai_pos_history_for_path_drawing.Count > MAX_POINTS) _ai_pos_history_for_path_drawing.Dequeue();
            }

            PointF current_pos_for_target_check = autoMove && _last_ai_pos.HasValue ? _last_ai_pos.Value : os_mouse_pos;
            if (Distance(current_pos_for_target_check, targetPos) < TARGET_RADIUS)
            {
                targetPos = RandomTarget();
            }

            if (TRAINING_MODE)
            {
                if (mouseTrail.Count >= 2)
                {
                    PointF current_for_tgt = mouseTrail.ElementAt(mouseTrail.Count - 1);
                    PointF prev_for_tgt = mouseTrail.ElementAt(mouseTrail.Count - 2);
                    CollectSample(current_for_tgt, prev_for_tgt, mouseTrail.ToList(), targetPos);
                }
            }
            else if (autoMove)
            {
                if (mouseTrail.Count >= HISTORY_POINTS_NEEDED_FOR_FEATURES)
                {
                    List<PointF> trail_for_prediction = _ai_pos_history_for_path_drawing.Count >= HISTORY_POINTS_NEEDED_FOR_FEATURES
                        ? _ai_pos_history_for_path_drawing.ToList()
                        : mouseTrail.ToList();

                    var next_predicted_pos = PredictNextPoint(trail_for_prediction, targetPos);
                    Cursor.Position = PointToScreen(Point.Round(next_predicted_pos));
                    _last_ai_pos = next_predicted_pos;
                }
            }
            else
            {
                _last_ai_pos = null;
                _ai_pos_history_for_path_drawing.Clear();
                _ai_pos_history_for_path_drawing.Enqueue(os_mouse_pos);
            }

            frameCounter++;
            if (fpsWatch.ElapsedMilliseconds >= 1000)
            {
                if (!trainingInProgress) Text = $"Fare Tahmincisi (Gelişmiş LSTM) – FPS: {frameCounter}";
                frameCounter = 0;
                fpsWatch.Restart();
            }
        }

        void CollectSample(PointF current_mouse_pos, PointF prev_mouse_pos, IReadOnlyList<PointF> trail_for_features, PointF currentTargetPos)
        {
            if (trail_for_features.Count < MOUSE_HISTORY_SEQUENCE_LENGTH + HISTORY_POINTS_NEEDED_FOR_FEATURES) return;

            var feats_sequence = PrepareFeatureSequence(trail_for_features, currentTargetPos);

            double[] dist_to_target_vec = { currentTargetPos.X - current_mouse_pos.X, currentTargetPos.Y - current_mouse_pos.Y };
            double target_dir_mag = Math.Sqrt(dist_to_target_vec[0] * dist_to_target_vec[0] + dist_to_target_vec[1] * dist_to_target_vec[1]);
            double target_ideal_dir_x = (target_dir_mag > 0.001) ? dist_to_target_vec[0] / target_dir_mag : 0;
            double target_ideal_dir_y = (target_dir_mag > 0.001) ? dist_to_target_vec[1] / target_dir_mag : 0;

            double[] last_move_vec = { current_mouse_pos.X - prev_mouse_pos.X, current_mouse_pos.Y - prev_mouse_pos.Y };
            double last_move_mag = Math.Sqrt(last_move_vec[0] * last_move_vec[0] + last_move_vec[1] * last_move_vec[1]);

            double speed_ratio_to_base = (MOUSE_STEP_SIZE > 0.001) ? last_move_mag / MOUSE_STEP_SIZE : 0;
            double tanh_target_speed = (2.0 * Math.Min(Math.Max(0, speed_ratio_to_base), MAX_SPEED_MULTIPLIER) / MAX_SPEED_MULTIPLIER) - 1.0;

            var tgt = new[] { target_ideal_dir_x, target_ideal_dir_y, Math.Max(-1.0, Math.Min(1.0, tanh_target_speed)) };

            if (last_move_mag > 0.1)
            {
                trainingData.Add((feats_sequence, tgt));
            }
        }

        PointF PredictNextPoint(IReadOnlyList<PointF> trail, PointF currentTargetPos)
        {
            if (trail.Count < MOUSE_HISTORY_SEQUENCE_LENGTH) return trail.LastOrDefault();

            var feats_for_pred = PrepareFeatureSequence(trail, currentTargetPos);
            var pred_output = net.Forward(feats_for_pred);

            double predicted_dir_x_raw = pred_output[0];
            double predicted_dir_y_raw = pred_output[1];
            double predicted_tanh_speed = pred_output[2];

            double pred_dir_mag = Math.Sqrt(predicted_dir_x_raw * predicted_dir_x_raw + predicted_dir_y_raw * predicted_dir_y_raw);
            double final_dir_x = (pred_dir_mag > 0.001) ? predicted_dir_x_raw / pred_dir_mag : 0;
            double final_dir_y = (pred_dir_mag > 0.001) ? predicted_dir_y_raw / pred_dir_mag : 0;

            double speed_multiplier_01 = (predicted_tanh_speed + 1.0) / 2.0;
            double actual_speed_multiplier = MIN_SPEED_MULTIPLIER + speed_multiplier_01 * (MAX_SPEED_MULTIPLIER - MIN_SPEED_MULTIPLIER);
            double actual_step_magnitude = actual_speed_multiplier * MOUSE_STEP_SIZE;

            PointF current_pos_from_trail = trail.Last();
            var next_pos = new PointF(
                (float)(current_pos_from_trail.X + final_dir_x * actual_step_magnitude),
                (float)(current_pos_from_trail.Y + final_dir_y * actual_step_magnitude));

            return new PointF(Math.Max(0, Math.Min(WIDTH, next_pos.X)), Math.Max(0, Math.Min(HEIGHT, next_pos.Y)));
        }

        void RunTraining()
        {
            if (trainingData.Count < BATCH_SIZE)
            {
                BeginInvoke((Action)(() => ShowMessage($"Eğitilecek yeterli veri yok! En az {BATCH_SIZE} örnek gerekiyor.")));
                trainingInProgress = false;
                return;
            }

            int epochs = 20;

            BeginInvoke((Action)(() =>
            {
                trainingProgressBar.Value = 0;
                trainingProgressBar.Visible = true;
                ShowMessage($"Eğitim başlıyor... Örnek: {trainingData.Count}, Tur: {epochs}, Yığın Boyutu: {BATCH_SIZE}");
            }));

            net.Train(trainingData, epochs, (ep, loss) =>
            {
                int percentage = (int)(((double)(ep + 1) / epochs) * 100);
                Console.Write($"\rEpoch {ep + 1}/{epochs} [{new string('#', percentage / 5).PadRight(20)}] {percentage,3}% - Loss: {loss:F8}      ");
                BeginInvoke((Action)(() =>
                {
                    if (trainingProgressBar.IsDisposed) return;
                    trainingProgressBar.Value = percentage;
                    Text = $"Eğitim... ({percentage}%) – Loss: {loss:F8}";
                }));
            });

            Console.WriteLine("\nEğitim tamamlandı.");
            net.Save("mouse_predictor_lstm.json");

            BeginInvoke((Action)(() =>
            {
                if (this.IsDisposed) return;
                trainingProgressBar.Visible = false;
                Text = "Fare Tahmincisi (Gelişmiş LSTM)";
                ShowMessage("Eğitim bitti ve model 'mouse_predictor_lstm.json' olarak kaydedildi!");
                trainingInProgress = false;
                trainingData.Clear();
            }));
        }

        #endregion

        #region Çizim ve Kullanıcı Etkileşimi

        void OnPaintCanvas(object sender, PaintEventArgs e)
        {
            var g = e.Graphics;
            g.Clear(Color.FromArgb(20, 20, 20));

            using (var redBrush = new SolidBrush(Color.Tomato))
                g.FillEllipse(redBrush, targetPos.X - TARGET_RADIUS, targetPos.Y - TARGET_RADIUS, TARGET_RADIUS * 2, TARGET_RADIUS * 2);

            PointF actual_cursor_pos = PointToClient(Cursor.Position);
            using (var blueBrush = new SolidBrush(Color.SkyBlue))
                g.FillEllipse(blueBrush, actual_cursor_pos.X - 7, actual_cursor_pos.Y - 7, 14, 14);

            if (autoMove && _last_ai_pos.HasValue)
            {
                using (var aiCursorBrush = new SolidBrush(Color.LimeGreen))
                    g.FillEllipse(aiCursorBrush, _last_ai_pos.Value.X - 5, _last_ai_pos.Value.Y - 5, 10, 10);
            }

            if ((TRAINING_MODE || autoMove) && mouseTrail.Count > 1)
            {
                using (var trailPen = new Pen(autoMove ? Color.FromArgb(100, 0, 200, 0) : Color.FromArgb(100, 200, 200, 0), 2))
                {
                    var history = autoMove ? _ai_pos_history_for_path_drawing.ToArray() : mouseTrail.ToArray();
                    if (history.Length > 1)
                        g.DrawLines(trailPen, history);
                }
            }
        }
        void OnKeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.V && !TRAINING_MODE)
            {
                autoMove = !autoMove;
                if (autoMove)
                {
                    _last_ai_pos = PointToClient(Cursor.Position);
                    _ai_pos_history_for_path_drawing.Clear();
                    for (int i = 0; i < MOUSE_HISTORY_SEQUENCE_LENGTH; ++i)
                        _ai_pos_history_for_path_drawing.Enqueue(_last_ai_pos.Value);
                }
            }
            if (e.KeyCode == Keys.S && TRAINING_MODE && trainingData.Count > 0 && !trainingInProgress)
            {
                trainingInProgress = true;
                Task.Run(() => RunTraining());
            }
        }
    }
}
