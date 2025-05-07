window.module2Data = [
  {
    id: 2,
    title: "Building Neural Networks",
    description:
      "This module transitions from basic tensor operations to constructing neural networks. Learn how to build, structure, and understand neural networks using PyTorch's nn module, including layers, activations, and model containers.",
    lessons: [
      {
        id: 1,
        title: "The nn.Module Class",
        content: `
          <h2>The nn.Module Class: Building Block of PyTorch Models</h2>
          <p>
            Now that you've learned about tensors and autograd, it's time to start building actual neural networks! The foundation of all neural networks in PyTorch is the <code>nn.Module</code> class.
          </p>

          <div class="lesson-section">
            <h3>What is nn.Module?</h3>
            <p>
              <code>nn.Module</code> is the base class for all neural network components in PyTorch. Whether you're creating:
            </p>
            <ul>
              <li>A single layer (like a linear/dense layer)</li>
              <li>An activation function</li>
              <li>A loss function</li>
              <li>Or an entire neural network</li>
            </ul>
            <p>
              They all inherit from this fundamental class. Think of it as the blueprint that gives all PyTorch models their core functionality.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Creating Your First Neural Network</h3>
            <p>
              To create a neural network in PyTorch, you'll define a new class that inherits from <code>nn.Module</code>. This class needs two key methods:
            </p>
            <ul>
              <li><strong>__init__(self)</strong> - Where you define the layers and components of your model</li>
              <li><strong>forward(self, x)</strong> - Where you specify how data flows through those layers</li>
            </ul>
            <p>Let's see a simple example:</p>
            
            <pre><code class="language-python">import torch
import torch.nn as nn  # The neural network module

# Define a simple model by inheriting from nn.Module
class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearModel, self).__init__()  # Always call the parent constructor first
        
        # Define the layers as attributes
        self.linear_layer = nn.Linear(input_size, output_size)
        
        print("SimpleLinearModel initialized.")
        print(f"  - Created linear layer: {self.linear_layer}")

    # Define the forward pass (how data flows through the model)
    def forward(self, x):
        # Pass the input tensor through our layer
        output = self.linear_layer(x)
        return output

# Create an instance of our model
input_dim = 10   # Input has 10 features
output_dim = 5   # Output has 5 features
model = SimpleLinearModel(input_dim, output_dim)

# Print the model structure
print("\\nModel Structure:")
print(model)

# Create a sample input batch (3 samples, 10 features each)
batch_size = 3
sample_input = torch.randn(batch_size, input_dim)
print(f"\\nInput shape: {sample_input.shape}")

# Pass the input through the model
output = model(sample_input)  # This calls model.forward() behind the scenes
print(f"Output shape: {output.shape}")
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Understanding Parameters</h3>
            <p>
              Neural networks have parameters (weights and biases) that need to be learned during training. When you define layers like <code>nn.Linear</code> inside an <code>nn.Module</code>, PyTorch automatically:
            </p>
            <ul>
              <li>Registers those parameters so they're tracked</li>
              <li>Makes them trainable by setting <code>requires_grad=True</code></li>
              <li>Initializes them with reasonable starting values</li>
            </ul>
            <p>Let's see how to access these parameters:</p>

            <pre><code class="language-python"># Print the model's parameters
print("\\nModel Parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
</code></pre>
            <p>
              For a linear layer with input size 10 and output size 5, we'd have:
            </p>
            <ul>
              <li>weight: shape=[5, 10] (5 outputs, each connected to 10 inputs)</li>
              <li>bias: shape=[5] (one bias per output)</li>
            </ul>
            <p>
              That's a total of 5×10 + 5 = 55 trainable parameters for this simple model!
            </p>
          </div>

          <div class="lesson-section">
            <h3>Why nn.Module is So Powerful</h3>
            <p>
              The <code>nn.Module</code> system gives you tremendous flexibility while handling lots of details automatically:
            </p>
            <ul>
              <li><strong>Parameter Management</strong> - Tracks and organizes trainable parameters</li>
              <li><strong>Device Support</strong> - Easily move your model between CPU and GPU</li>
              <li><strong>Saving & Loading</strong> - Simple saving and loading of model weights</li>
              <li><strong>Evaluation Mode</strong> - Switch between training and evaluation behaviors</li>
              <li><strong>Hierarchical Design</strong> - Build complex models by nesting simpler ones</li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Challenge:</strong> Time to build your first neural network! Define a class called <code>TwoLayerModel</code> that inherits from <code>nn.Module</code>. It should:
              </p>
              <ol>
                <li>Have an <code>__init__</code> method that takes <code>input_size</code>, <code>hidden_size</code>, and <code>output_size</code> as parameters</li>
                <li>Define two linear layers:
                  <ul>
                    <li><code>layer1</code>: from <code>input_size</code> to <code>hidden_size</code></li>
                    <li><code>layer2</code>: from <code>hidden_size</code> to <code>output_size</code></li>
                  </ul>
                </li>
                <li>Have a <code>forward</code> method that passes input through <code>layer1</code> then <code>layer2</code></li>
              </ol>
              <p>After defining it, create an instance with sizes 20, 10, and 2, print the model, and count its parameters.</p>

              <pre><code class="language-python">class TwoLayerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerModel, self).__init__()
        # Define your layers here
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Define the forward pass here
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Create your model with input_size=20, hidden_size=10, output_size=2
my_model = TwoLayerModel(20, 10, 2)

# Print the model structure
print(my_model)

# Count parameters
total_params = sum(p.numel() for p in my_model.parameters())
print(f"Total parameters: {total_params}")  # Should be 20*10 + 10 + 10*2 + 2 = 232
</code></pre>
            </div>
          </div>
        `,
        completed: false,
      },
      {
        id: 2,
        title: "Common Layers and Activation Functions",
        content: `
          <h2>Common Layers and Activation Functions</h2>
          <p>
            Now that you understand the basic structure of PyTorch models with <code>nn.Module</code>, let's explore the building blocks you'll use to create neural networks: layers and activation functions.
          </p>

          <div class="lesson-section">
            <h3>Essential Neural Network Layers</h3>
            <p>
              PyTorch provides many types of layers in the <code>torch.nn</code> module. Here are the most common ones you'll use:
            </p>
            
            <h4>Linear Layers (Fully Connected)</h4>
            <p>
              <code>nn.Linear</code> is the most basic layer type. It connects every input feature to every output feature with learned weights.
            </p>
            <pre><code class="language-python">import torch
import torch.nn as nn

# Linear layer: connects 10 input features to 5 output features
linear_layer = nn.Linear(in_features=10, out_features=5)

# Apply to an input (batch_size=3, features=10)
input_tensor = torch.randn(3, 10)
output_tensor = linear_layer(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")  # Should be [3, 5]
</code></pre>

            <h4>Convolutional Layers</h4>
            <p>
              <code>nn.Conv2d</code> applies learned filters to input images to detect features like edges, textures, and patterns. These are fundamental for image processing tasks.
            </p>
            <pre><code class="language-python"># 2D Convolutional Layer
# Arguments: (input_channels, output_channels, kernel_size)
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# Apply to an image batch (batch_size=2, channels=3, height=28, width=28)
image_batch = torch.randn(2, 3, 28, 28)
feature_maps = conv_layer(image_batch)
print(f"Image batch shape: {image_batch.shape}")
print(f"Feature maps shape: {feature_maps.shape}")  # Should be [2, 16, 28, 28]
</code></pre>
            <p>
              We'll explore convolutional networks in depth in later modules. For now, just understand they're specialized for grid-like data like images.
            </p>
            
            <h4>Recurrent Layers</h4>
            <p>
              <code>nn.LSTM</code> and <code>nn.GRU</code> process sequential data (like text or time series) by maintaining internal memory states. They're designed to capture patterns over sequences.
            </p>
            <p>
              We'll cover these in more detail in our advanced modules. Let's focus on feed-forward networks for now.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Activation Functions: Adding Non-Linearity</h3>
            <p>
              Activation functions are crucial in neural networks. Without them, stacking multiple linear layers would just be equivalent to a single linear layer! 
            </p>
            
            <p>
              <strong>Why?</strong> Because combining linear operations (like matrix multiplications) just gives you another linear operation. To learn complex patterns, networks need <em>non-linearity</em>.
            </p>
            
            <h4>Common Activation Functions</h4>
            
            <ul>
              <li>
                <strong>ReLU</strong> (Rectified Linear Unit): <code>nn.ReLU()</code>
                <p>The most widely used activation function. It simply outputs the input if it's positive, otherwise outputs zero.</p>
                    <p><code>f(x) = max(0, x)</code></p>
                <p>Advantages: Fast to compute, helps prevent the "vanishing gradient" problem (where gradients become too small during backpropagation, making training ineffective).</p>
              </li>
              
              <li>
                <strong>Sigmoid</strong>: <code>nn.Sigmoid()</code>
                <p>Squashes values between 0 and 1, making it useful for binary classification outputs and gates.</p>
                <p><code>f(x) = 1 / (1 + e^(-x))</code></p>
                <p>Caution: Can suffer from vanishing gradients for extreme values.</p>
              </li>
              
              <li>
                <strong>Tanh</strong>: <code>nn.Tanh()</code>
                <p>Similar to sigmoid but outputs values between -1 and 1.</p>
                <p><code>f(x) = tanh(x)</code></p>
                <p>Often performs better than sigmoid in hidden layers, but still has vanishing gradient issues.</p>
              </li>
              
              <li>
                <strong>Softmax</strong>: <code>nn.Softmax(dim=1)</code>
                <p>Converts a vector of values into a probability distribution (values sum to 1).</p>
                <p>Typically used for multi-class classification in the output layer.</p>
                <p>The <code>dim</code> parameter specifies which dimension to apply softmax over (usually the class dimension).</p>
              </li>
            </ul>
            
            <pre><code class="language-python"># Let's see how ReLU transforms values
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
relu = nn.ReLU()
y = relu(x)
print(f"Input: {x}")
print(f"After ReLU: {y}")  # Should be [0, 0, 0, 1, 2]

# Sigmoid example
sigmoid = nn.Sigmoid()
y_sigmoid = sigmoid(x)
print(f"After Sigmoid: {y_sigmoid}")  # Values between 0 and 1

# Softmax example (converting scores to probabilities)
scores = torch.tensor([[2.0, 1.0, 0.1], [0.1, 5.0, 1.0]])
softmax = nn.Softmax(dim=1)  # Apply along dimension 1 (across columns)
probabilities = softmax(scores)
print(f"Raw scores:\\n{scores}")
print(f"Probabilities after Softmax:\\n{probabilities}")
print(f"Sum of each row (should be 1): {probabilities.sum(dim=1)}")
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>What's the "Vanishing Gradient" Problem?</h3>
            <p>
              You'll often hear this term in deep learning. The vanishing gradient problem happens when:
            </p>
            <ul>
              <li>Gradients (signals for how to update weights) become extremely small</li>
              <li>When they're small, weights get updated by tiny amounts</li>
              <li>This makes training very slow or gets stuck, especially in deeper networks</li>
            </ul>
            <p>
              It's like trying to tell someone to adjust their position, but your voice becomes so faint the further they are from you that they can't hear your instructions.
            </p>
            <p>
              Activation functions like ReLU help combat this problem by maintaining stronger gradient signals, which is why they've become so popular in modern networks.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Building Models with nn.Sequential</h3>
            <p>
              For simple feed-forward networks where data flows straight through each layer, PyTorch offers the <code>nn.Sequential</code> container as a shortcut.
            </p>
            
            <p>
              This saves you from having to write a custom <code>nn.Module</code> class with <code>__init__</code> and <code>forward</code> methods.
            </p>
            
            <pre><code class="language-python"># Building a simple feed-forward network with nn.Sequential
model = nn.Sequential(
    nn.Linear(784, 128),  # First layer (input → hidden)
    nn.ReLU(),            # Activation function
    nn.Linear(128, 64),   # Second layer (hidden → hidden)
    nn.ReLU(),            # Another activation
    nn.Linear(64, 10)     # Output layer (hidden → output)
)

print("Sequential model structure:")
print(model)

# Create dummy input (batch_size=5, input_features=784)
dummy_input = torch.randn(5, 784)

# Pass through the model
output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")  # Should be [5, 10]
</code></pre>
            
            <p>
              <strong>When to use Sequential?</strong> It's perfect for straightforward architectures where each layer feeds directly into the next. For more complex patterns (branching, skipping connections, etc.), you'll still need custom <code>nn.Module</code> classes.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Challenge:</strong> Create a neural network for image classification using <code>nn.Sequential</code>. The network should:
              </p>
              <ol>
                <li>Take inputs with 784 features (like flattened 28×28 MNIST images)</li>
                <li>Include three linear layers with dimensions:
                  <ul>
                    <li>784 → 128</li>
                    <li>128 → 64</li> 
                    <li>64 → 10 (for 10 classes like digits 0-9)</li>
                  </ul>
                </li>
                <li>Include ReLU activations after the first and second linear layers</li>
                <li>Process a dummy batch of 3 images and print the output shape</li>
              </ol>

              <pre><code class="language-python">import torch
import torch.nn as nn

# Create your sequential model here
mnist_model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Print the model structure
print(mnist_model)

# Create a batch of 3 dummy images (flattened 28×28 images)
dummy_images = torch.randn(3, 784)

# Get predictions
with torch.no_grad():  # We don't need gradients for a simple forward pass
    predictions = mnist_model(dummy_images)

print(f"Input shape: {dummy_images.shape}")
print(f"Output shape: {predictions.shape}")  # Should be [3, 10]
</code></pre>
            </div>
          </div>
        `,
        completed: false,
      },
      {
        id: 3,
        title: "Loss Functions",
        content: `
          <h2>Loss Functions: Measuring Model Performance</h2>
          <p>
            After building our neural network, we need a way to measure how well (or poorly) it's performing. This is where <strong>loss functions</strong> come in - they tell us how far off our model's predictions are from the true values.
          </p>

          <div class="lesson-section">
            <h3>What Are Loss Functions?</h3>
            <p>
              A loss function (also called a cost function or criterion) computes a single value that represents how "wrong" our model's predictions are compared to the actual target values. The lower this value, the better our model is performing.
            </p>
            <p>
              Think of it like a score in a game where you're trying to get the <em>lowest</em> possible score. During training, we'll try to minimize this value by adjusting our model parameters.
            </p>
            <p>
              PyTorch provides a variety of loss functions in the <code>torch.nn</code> module, each designed for different types of problems.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Common Loss Functions</h3>
            
            <h4>For Regression (Predicting Continuous Values)</h4>
            <p>
              <strong>Mean Squared Error (MSELoss)</strong> - Calculates the average of squared differences between predictions and targets. It heavily penalizes large errors because of the squaring.
            </p>
            <pre><code class="language-python">import torch
import torch.nn as nn

# Create MSE loss function
loss_fn = nn.MSELoss()

# Predictions (outputs from your model) - batch of 3 samples, each with 1 value
predictions = torch.tensor([[1.5], [3.8], [2.1]], dtype=torch.float32)
# Actual target values
targets = torch.tensor([[1.0], [4.0], [2.0]], dtype=torch.float32)

# Calculate loss
loss = loss_fn(predictions, targets)
print(f"Predictions: {predictions.squeeze()}")
print(f"Targets: {targets.squeeze()}")
print(f"MSE Loss: {loss.item()}")  # Should be close to 0.043
</code></pre>

            <h4>For Classification (Predicting Categories)</h4>
            <p>
              <strong>Cross Entropy Loss (CrossEntropyLoss)</strong> - Used for multi-class classification problems (like classifying digits 0-9). It combines softmax activation and negative log likelihood loss.
            </p>
            <pre><code class="language-python"># Create Cross Entropy loss function
loss_fn = nn.CrossEntropyLoss()

# Predictions (raw scores/logits from your model) - batch of 3 samples, 5 classes each
predictions = torch.tensor([
    [2.0, 1.0, 0.1, 0.7, 0.2],  # Sample 1: highest score for class 0
    [0.2, 3.0, 1.1, 0.7, 0.1],  # Sample 2: highest score for class 1
    [0.3, 0.5, 0.2, 0.8, 2.2]   # Sample 3: highest score for class 4
], dtype=torch.float32)

# Actual class labels (integers representing the correct class)
targets = torch.tensor([0, 1, 4])  # Classes start from 0

# Calculate loss
loss = loss_fn(predictions, targets)
print(f"Cross Entropy Loss: {loss.item()}")
</code></pre>
            <p>
              <strong>Important:</strong> For <code>CrossEntropyLoss</code>, you provide raw model outputs (logits), not softmax probabilities. The loss function applies softmax internally.
            </p>

            <h4>For Binary Classification (Yes/No Problems)</h4>
            <p>
              <strong>Binary Cross Entropy with Logits (BCEWithLogitsLoss)</strong> - Used when classifying into just two categories, or when each sample can have multiple labels. It combines sigmoid activation with binary cross entropy loss.
            </p>
            <pre><code class="language-python"># Create BCE with Logits loss function
loss_fn = nn.BCEWithLogitsLoss()

# Predictions (raw scores from your model) - batch of 2 samples, 3 binary labels each
predictions = torch.tensor([
    [2.0, -1.0, 0.5],   # Sample 1: predicted scores for 3 binary labels
    [-0.5, 1.2, 0.8]    # Sample 2: predicted scores for 3 binary labels
], dtype=torch.float32)

# Target values (1.0 = positive, 0.0 = negative)
targets = torch.tensor([
    [1.0, 0.0, 1.0],    # Sample 1: first and third labels are positive
    [0.0, 1.0, 1.0]     # Sample 2: second and third labels are positive
], dtype=torch.float32)

# Calculate loss
loss = loss_fn(predictions, targets)
print(f"BCE With Logits Loss: {loss.item()}")
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Loss Functions Flow in Training</h3>
            <p>
              Here's how loss functions fit into the overall training process:
            </p>
            <ol>
              <li>Your model makes predictions</li>
              <li>You calculate the loss between predictions and actual targets</li>
              <li>This loss value guides the parameter updates during backpropagation</li>
              <li>Your goal is to minimize this loss value by adjusting model parameters</li>
            </ol>
            <p>
              The loss gives your model feedback on how well it's doing and which direction to adjust its parameters.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Choosing the Right Loss Function</h3>
            <p>
              The type of problem you're solving determines which loss function to use:
            </p>
            <ul>
              <li><strong>Regression</strong> (predicting continuous values like price, temperature, etc.) → <code>nn.MSELoss</code> or <code>nn.L1Loss</code></li>
              <li><strong>Multi-class Classification</strong> (predicting one class from multiple options) → <code>nn.CrossEntropyLoss</code></li>
              <li><strong>Binary Classification</strong> (yes/no problems) → <code>nn.BCEWithLogitsLoss</code></li>
              <li><strong>Multi-label Classification</strong> (items can belong to multiple classes) → <code>nn.BCEWithLogitsLoss</code></li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Challenge:</strong> Imagine you have a multi-class classification problem with 4 classes. Your model outputs the following <strong>logits</strong> for a batch of 2 samples:
              </p>
              <pre><code class="language-python">import torch
import torch.nn as nn

# Model outputs logits (raw scores) for 4 classes
model_output = torch.tensor([[1.2, 0.1, -0.4, 1.5], [-0.1, 2.0, 0.5, 0.3]])

# True class indices for these samples (classes are 0-indexed)
true_labels = torch.tensor([3, 1])  # First sample is class 3, second is class 1

# 1. Which loss function is appropriate here? (Answer: CrossEntropyLoss)

# 2. Instantiate the appropriate loss function
loss_fn = nn.CrossEntropyLoss()

# 3. Calculate the loss
loss = loss_fn(model_output, true_labels)
print(f"Loss: {loss.item()}")
</code></pre>
              <p>
                <strong>Question:</strong> Why is <code>nn.MSELoss</code> <em>not</em> appropriate for this task?
              </p>
            </div>
          </div>
        `,
        completed: false,
      },
      {
        id: 4,
        title: "Optimizers",
        content: `
          <h2>Optimizers: Teaching Your Network to Learn</h2>
          <p>
            Now that we can measure how wrong our model's predictions are using loss functions, we need a way to make our model better. <strong>Optimizers</strong> are algorithms that adjust our model's parameters (weights and biases) to reduce the loss.
          </p>

          <div class="lesson-section">
            <h3>What Are Optimizers?</h3>
            <p>
              Optimizers implement different strategies for updating model parameters based on the gradients computed during backpropagation. Their job is to find the best values for all the weights and biases in your network.
            </p>
            <p>
              Think of training as climbing down a mountain (where the height represents the loss). The optimizer determines:
            </p>
            <ul>
              <li>Which direction to step (using gradients)</li>
              <li>How big each step should be (learning rate)</li>
              <li>Whether to adjust the step size over time</li>
              <li>Whether to consider previous steps when deciding the next one</li>
            </ul>
            <p>
              PyTorch provides various optimizers in the <code>torch.optim</code> module.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Common Optimizers</h3>
            
            <h4>Stochastic Gradient Descent (SGD)</h4>
            <p>
              The simplest optimization algorithm. It updates parameters in the opposite direction of their gradients, scaled by a learning rate.
            </p>
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple model
model = nn.Linear(10, 2)  # 10 input features, 2 output features

# Create an SGD optimizer
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print("SGD Optimizer created with learning rate:", learning_rate)
</code></pre>

            <h4>Adam (Adaptive Moment Estimation)</h4>
            <p>
              A more sophisticated optimizer that adapts the learning rate for each parameter individually. It often works better than SGD and requires less tuning.
            </p>
            <pre><code class="language-python"># Create an Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Adam Optimizer created")
</code></pre>
            <p>
              Adam is a great default choice for many problems, as it typically converges faster than SGD.
            </p>
          </div>

          <div class="lesson-section">
            <h3>The Optimization Process</h3>
            <p>
              The core optimization routine always follows these three steps:
            </p>
            <ol>
              <li>
                <strong>Zero the gradients</strong> - Call <code>optimizer.zero_grad()</code> to clear any existing gradients
              </li>
              <li>
                <strong>Compute the loss and gradients</strong> - Call <code>loss.backward()</code> to calculate gradients
              </li>
              <li>
                <strong>Update the parameters</strong> - Call <code>optimizer.step()</code> to update weights and biases
              </li>
            </ol>
            <p>
              Let's see this in action:
            </p>
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple model
model = nn.Linear(10, 2)

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Simulate some training data
inputs = torch.randn(5, 10)  # 5 samples, 10 features each
targets = torch.randn(5, 2)  # 5 samples, 2 target values each

# Choose a loss function
loss_fn = nn.MSELoss()

# Store initial weights for comparison
old_weight = model.weight.clone().detach()

# --- THE CORE OPTIMIZATION LOOP ---
# Step 1: Zero the gradients (always do this first!)
optimizer.zero_grad()

# Step 2a: Forward pass - get predictions
outputs = model(inputs)

# Step 2b: Calculate loss
loss = loss_fn(outputs, targets)
print(f"Loss: {loss.item():.4f}")

# Step 3: Backward pass - compute gradients
loss.backward()

# Step 4: Update parameters using optimizer
optimizer.step()

# Check if weights changed
print("Did weights change?", not torch.equal(old_weight, model.weight))
</code></pre>
            <p>
              <strong>Important note:</strong> Always call <code>optimizer.zero_grad()</code> before calculating new gradients! Otherwise, PyTorch will accumulate gradients from previous steps, which is usually not what you want.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Learning Rate</h3>
            <p>
              The <code>lr</code> parameter (learning rate) is the most critical hyperparameter for optimizers. It controls how big the steps are when updating model parameters:
            </p>
            <ul>
              <li><strong>Too large:</strong> The model might overshoot the optimal values and fail to converge</li>
              <li><strong>Too small:</strong> The model will learn very slowly and might get stuck in suboptimal areas</li>
            </ul>
            <p>
              Starting with a learning rate between 0.1 and 0.0001 is usually reasonable, then adjusting based on results.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Putting It All Together</h3>
            <p>
              Here's a complete example showing a single training step:
            </p>
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim

# Create a small model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)
        self.activation = nn.ReLU()
        self.output = nn.Linear(5, 1)
        
    def forward(self, x):
        x = self.activation(self.layer(x))
        return self.output(x)

# Instantiate model
model = SimpleModel()

# Create optimizer 
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create dummy data
inputs = torch.randn(8, 10)  # 8 samples, 10 features each
targets = torch.randn(8, 1)  # 8 samples, 1 target value each

# Choose loss function
loss_fn = nn.MSELoss()

# --- ONE TRAINING STEP ---
# Zero gradients
optimizer.zero_grad()

# Forward pass
outputs = model(inputs)

# Calculate loss
loss = loss_fn(outputs, targets)
print(f"Loss before training: {loss.item():.6f}")

# Backward pass (compute gradients)
loss.backward()

# Update parameters 
optimizer.step()

# Check if model improved
outputs_after = model(inputs)
loss_after = loss_fn(outputs_after, targets)
print(f"Loss after one step: {loss_after.item():.6f}")
print(f"Did loss decrease? {loss_after < loss}")
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Challenge:</strong> Train a TwoLayerModel for one step using SGD optimizer.
              </p>
              <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim

# Recreate the TwoLayerModel from Lesson 1
class TwoLayerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Create model instance
model = TwoLayerModel(input_size=20, hidden_size=10, output_size=2)

# Create dummy data
batch_size = 5
x = torch.randn(batch_size, 20)  # 5 samples, 20 features each
y = torch.randn(batch_size, 2)   # 5 samples, 2 target values each

# Choose a loss function for regression
loss_fn = nn.MSELoss()

# Create an SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ====== YOUR OPTIMIZATION STEP ======
# 1. Zero gradients
optimizer.zero_grad()

# 2. Forward pass
predictions = model(x)

# 3. Calculate loss
loss = loss_fn(predictions, y)
print(f"Initial loss: {loss.item():.4f}")

# 4. Backward pass
loss.backward()

# 5. Update parameters
optimizer.step()

# Check if model improved
with torch.no_grad():  # Don't track gradients for this evaluation
    new_predictions = model(x)
    new_loss = loss_fn(new_predictions, y)
    print(f"Loss after one step: {new_loss.item():.4f}")
    print(f"Did loss decrease? {new_loss < loss}")
</code></pre>
            </div>
          </div>
        `,
        completed: false,
      },
      {
        id: 5,
        title: "Module 2 Practice Task",
        content: `
          <h2>Module 2 Cumulative Task: Mini Training Step Simulation</h2>
          <p>
            Now that you've learned about neural network modules, layers, loss functions, and optimizers, it's time to put it all together! In this practice task, you'll simulate a complete training step for a simple regression model.
          </p>

          <div class="lesson-section">
            <h3>Objective</h3>
            <p>
              Simulate a single forward and backward pass with parameter updates for a simple regression model, applying all the core concepts from this module.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Part 1: Define the Model</h3>
            <p>
              First, let's create a custom neural network by applying what you learned in Lessons 2.1 and 2.2. We'll build a simple regression model with two linear layers and a ReLU activation.
            </p>
            
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim

# Define a custom neural network for regression
class SimpleRegressionModel(nn.Module):
    def __init__(self):
        super(SimpleRegressionModel, self).__init__()
        # Linear layer: 15 input features → 10 hidden features
        self.layer1 = nn.Linear(15, 10)
        # ReLU activation function
        self.activation = nn.ReLU()
        # Linear layer: 10 hidden features → 1 output feature
        self.layer2 = nn.Linear(10, 1)
    
    def forward(self, x):
        # Pass input through first layer
        x = self.layer1(x)
        # Apply activation function
        x = self.activation(x)
        # Pass through second layer to produce final output
        x = self.layer2(x)
        return x</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Part 2: Instantiate Components</h3>
            <p>
              Now let's create instances of our model, an appropriate loss function for regression, and an optimizer:
            </p>
            
            <pre><code class="language-python"># Create an instance of your model
model = SimpleRegressionModel()

# Define a loss function for regression (from Lesson 2.3)
# MSE is appropriate for regression problems
loss_function = nn.MSELoss()

# Choose an optimizer (from Lesson 2.4)
# Let's use Adam with a learning rate of 0.005
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Print the components we've created
print("Model architecture:")
print(model)
print("\nLoss function:", loss_function)
print("Optimizer:", optimizer)</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Part 3: Prepare Data</h3>
            <p>
              Let's create some dummy input and target tensors to use for our training simulation:
            </p>
            
            <pre><code class="language-python"># Create a batch of 8 samples with 15 features each
input_data = torch.randn(8, 15)

# Create corresponding target values (8 samples, 1 output each)
target_data = torch.randn(8, 1)

print(f"Input shape: {input_data.shape}")
print(f"Target shape: {target_data.shape}")</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Part 4: Perform One Training Step</h3>
            <p>
              Now we'll execute a complete training step by following the core optimization routine you learned in Lesson 2.4:
            </p>
            
            <pre><code class="language-python"># Step 1: Zero gradients
optimizer.zero_grad()

# Step 2: Forward pass - get predictions from model
predictions = model(input_data)

# Step 3: Calculate loss
loss = loss_function(predictions, target_data)

# Step 4: Backward pass - calculate gradients
loss.backward()

# Step 5: Update weights
optimizer.step()

# Print the loss value
print(f"Loss after one training step: {loss.item():.6f}")</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Verifying Our Progress</h3>
            <p>
              Let's check if our model improved by calculating the loss again with the updated weights:
            </p>
            
            <pre><code class="language-python"># Re-evaluate the model with the same data
with torch.no_grad():  # No need to track gradients for this evaluation
    new_predictions = model(input_data)
    new_loss = loss_function(new_predictions, target_data)
    
print(f"Original loss: {loss.item():.6f}")
print(f"New loss: {new_loss.item():.6f}")
print(f"Did loss decrease? {'Yes' if new_loss < loss else 'No'}")</code></pre>

            <p>
              If your implementation is correct, you should see that the loss has decreased after the parameter update, indicating that your model has learned something from this single training step!
            </p>
          </div>

          <div class="lesson-section">
            <h3>Complete Solution</h3>
            <p>
              Here's the complete code that combines all the steps above:
            </p>
            
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class SimpleRegressionModel(nn.Module):
    def __init__(self):
        super(SimpleRegressionModel, self).__init__()
        self.layer1 = nn.Linear(15, 10)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Instantiate components
model = SimpleRegressionModel()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Prepare data
input_data = torch.randn(8, 15)
target_data = torch.randn(8, 1)

# Perform one training step
optimizer.zero_grad()
predictions = model(input_data)
loss = loss_function(predictions, target_data)
loss.backward()
optimizer.step()

# Show result
print(f"Loss: {loss.item():.6f}")

# Verify improvement
with torch.no_grad():
    new_predictions = model(input_data)
    new_loss = loss_function(new_predictions, target_data)
    print(f"New loss: {new_loss.item():.6f}")
    print(f"Improvement: {loss.item() - new_loss.item():.6f}")</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Challenge: Extend Your Learning</h3>
            <div class="challenge-box">
              <p>
                <strong>Try these extensions</strong> to deepen your understanding:
              </p>
              <ol>
                <li>Add a third linear layer to the model and observe how the architecture changes</li>
                <li>Train the model for multiple steps in a loop and plot the loss values to see the learning curve</li>
                <li>Compare different optimizers (SGD vs. Adam) and different learning rates</li>
                <li>Create separate training and validation data to monitor for overfitting</li>
              </ol>
            </div>
          </div>

          <div class="lesson-section">
            <h3>Congratulations!</h3>
            <p>
              You've successfully completed Module 2! You now understand how to:
            </p>
            <ul>
              <li>Create custom neural networks with <code>nn.Module</code></li>
              <li>Use different layer types and activation functions</li>
              <li>Choose appropriate loss functions for your tasks</li>
              <li>Implement the complete training step with optimizers</li>
            </ul>
            <p>
              With these fundamental building blocks, you're ready to create and train more complex neural networks for real-world applications!
            </p>
          </div>
        `,
        completed: false,
      },
    ],
  },
];
