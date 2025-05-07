window.module3Data = [
  {
    id: 3,
    title: "Neural Networks",
    description:
      "Build and train basic neural networks using PyTorch's nn module.",
    lessons: [
      {
        id: 1,
        title: "Datasets and DataLoaders",
        content: `
          <h2>Datasets and DataLoaders: Efficiently Handling Data for Training</h2>
          <p>
            In this lesson, we'll learn how to efficiently load, preprocess, and iterate over data in batches using PyTorch's <code>Dataset</code> and <code>DataLoader</code> utilities.
          </p>

          <div class="lesson-section">
            <h3>Why Data Handling is Important</h3>
            <p>
              Training deep learning models often involves large datasets that don't fit into memory all at once. We need efficient ways to:
            </p>
            <ul>
              <li>Load data in small batches</li>
              <li>Shuffle it randomly during training (to prevent the model from learning the order of data)</li>
              <li>Preprocess it (e.g., convert images to tensors, normalize pixel values)</li>
            </ul>
            <p>
              Doing this manually can be complex and slow. PyTorch provides tools that make this process much easier.
            </p>
          </div>

          <div class="lesson-section">
            <h3>The Dataset Class</h3>
            <p>
              <code>torch.utils.data.Dataset</code> is an abstract class representing a dataset. Any custom dataset you create should inherit from this class. The two essential methods you <em>must</em> implement are:
            </p>
            <ul>
              <li><strong>__len__(self)</strong>: Should return the total number of samples in the dataset.</li>
              <li><strong>__getitem__(self, idx)</strong>: Should return the data sample and its corresponding label at the given index.</li>
            </ul>
            <p>
              Thankfully, for many common datasets like MNIST, CIFAR-10, ImageNet, etc., PyTorch's <code>torchvision</code> library provides ready-to-use <code>Dataset</code> subclasses. This saves you the effort of downloading and writing parsing code yourself.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Transforms for Preprocessing</h3>
            <p>
              Datasets often require preprocessing. Images need to be converted into tensors, and pixel values are usually normalized. <code>torchvision.transforms</code> provides common image transformation functions:
            </p>
            <ul>
              <li><strong>transforms.ToTensor()</strong>: Converts a PIL Image or NumPy array into a PyTorch tensor with values scaled to [0.0, 1.0].</li>
              <li><strong>transforms.Normalize(mean, std)</strong>: Normalizes a tensor image using the provided mean and standard deviation.</li>
              <li><strong>transforms.Compose([...])</strong>: Chains multiple transforms together into a single pipeline.</li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>DataLoader: Batching and Iteration</h3>
            <p>
              <code>torch.utils.data.DataLoader</code> is the workhorse for loading data during training/evaluation. It wraps a <code>Dataset</code> and provides an iterator that yields batches of data. Key functionalities:
            </p>
            <ul>
              <li><strong>batch_size</strong>: The number of samples per batch.</li>
              <li><strong>shuffle</strong>: If <code>True</code>, the data is reshuffled at every epoch. Crucial for training.</li>
              <li><strong>num_workers</strong>: Number of subprocesses to use for data loading. Setting this to > 0 can speed up data loading.</li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>Loading the MNIST Dataset</h3>
            <p>Let's see how to load the MNIST dataset, a classic "hello world" for image classification:</p>
            
            <p>First, we'll import the necessary libraries and set up our transformation pipeline:</p>
            <p>Define transformations for the MNIST dataset:</p>
            <ol>
              <li>Convert image to PyTorch Tensor (scales pixels to [0, 1])</li>
              <li>Normalize tensor image with mean and std dev of MNIST dataset</li>
            </ol>
            
            <pre><code class="language-python">import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std deviation (single channel)
])</code></pre>

            <p>Next, we'll load the MNIST training dataset. The key parameters are:</p>
            <ul>
              <li><strong>root</strong>: directory where data will be stored/downloaded</li>
              <li><strong>train=True</strong>: specifies to get the training set</li>
              <li><strong>download=True</strong>: downloads the data if not found in root directory</li>
              <li><strong>transform</strong>: applies the defined transformations to each image</li>
            </ul>

            <pre><code class="language-python">print("Downloading/Loading MNIST Training set...")
train_dataset = torchvision.datasets.MNIST(
    root='./data', # You can change this path
    train=True,
    download=True,
    transform=transform_pipeline
)
print(f"MNIST Training dataset loaded.")
print(f"Number of training samples: {len(train_dataset)}")

# Get one sample to inspect
img, label = train_dataset[0] # Calls __getitem__(0)
print(f"\\nShape of one image tensor: {img.shape}") # Shape: [Channel, Height, Width] -> [1, 28, 28] for MNIST
print(f"Label of the first image: {label}")
print(f"Min/Max pixel values after ToTensor & Normalize: {img.min()}, {img.max()}")</code></pre>

            <p>Finally, we'll create a DataLoader to handle batching and iterate through the first batch:</p>

            <pre><code class="language-python">batch_size = 64 # Number of samples per batch

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True, # Shuffle data at every epoch (important for training)
    num_workers=0 # Start with 0, increase if data loading is a bottleneck
)
print(f"\\nCreated DataLoader with batch size {batch_size} and shuffle=True.")

print("\\nIterating through the first batch:")
try:
    # Get the first batch
    first_batch = next(iter(train_loader))
    data_batch, target_batch = first_batch

    print(f"Data batch shape: {data_batch.shape}")  # Expected: [batch_size, channels, height, width] -> [64, 1, 28, 28]
    print(f"Target batch shape: {target_batch.shape}")  # Expected: [batch_size] -> [64]
    print(f"Target batch (first 10 labels): {target_batch[:10]}")

except Exception as e:
    print(f"Error retrieving batch: {e}")

print("\\nLesson 3.1 complete.")</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Task 3.1:</strong> Your turn! Using the concepts and code structure above:
              </p>
              <ol>
                <li>Load the MNIST <strong>test</strong> dataset using <code>torchvision.datasets.MNIST</code>. Remember to set <code>train=False</code> and apply the <em>same</em> <code>transform_pipeline</code>.</li>
                <li>Create a <code>DataLoader</code> called <code>test_loader</code> for this test dataset. Use a <code>batch_size</code> of 1000 and set <code>shuffle=False</code>.</li>
                <li>Iterate through the <code>test_loader</code> to get the <em>first batch</em> of test data and targets.</li>
                <li>Print the shapes of the data tensor and the target tensor for this first test batch.</li>
              </ol>

              <pre><code class="language-python"># Your code here to load the MNIST test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    download=True,
    transform=transform_pipeline
)
print(f"Number of test samples: {len(test_dataset)}")

# Create a DataLoader for the test dataset
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=0
)

# Get the first batch of test data
test_batch = next(iter(test_loader))
test_data, test_targets = test_batch

# Print shapes
print(f"Test data shape: {test_data.shape}")  # Should be [1000, 1, 28, 28]
print(f"Test targets shape: {test_targets.shape}")  # Should be [1000]
</code></pre>
            </div>
          </div>
        `,
        completed: false,
      },
      {
        id: 2,
        title: "Putting It All Together: A Basic Training Loop",
        content: `
          <h2>Putting It All Together: A Basic Training Loop</h2>
          <p>
            Now that we know how to load and preprocess data, it's time to write a complete pipeline to train a neural network. In this lesson, we'll implement a training loop for a simple model on the MNIST dataset.
          </p>

          <div class="lesson-section">
            <h3>Key Training Concepts</h3>
            <p>
              Before diving into code, let's understand some key concepts:
            </p>
            <ul>
              <li><strong>Epoch</strong>: One full pass through the entire training dataset. For example, if your dataset has 10,000 images and your batch size is 100, one epoch will involve 100 iterations (batches).</li>
              <li><strong>Batch</strong>: A small subset of the dataset processed at one time. The DataLoader provides these batches.</li>
              <li><strong>Training Mode</strong>: Some layers like Dropout and BatchNorm behave differently during training versus evaluation. The <code>model.train()</code> method ensures they're in the correct mode.</li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>The Training Loop Structure</h3>
            <p>
              A typical training loop involves an outer loop for epochs and an inner loop for batches within each epoch:
            </p>
            
            <h4>Outer Loop (Epochs)</h4>
            <pre><code class="language-python">for epoch in range(num_epochs):
    # ... training for one epoch ...</code></pre>
            
            <h4>Inner Loop (Batches)</h4>
            <p>Inside the epoch loop, you iterate over the DataLoader:</p>
            <pre><code class="language-python">for batch_idx, (data, targets) in enumerate(train_loader):
    # ... process one batch ...</code></pre>
            
            <h4>Steps Inside the Batch Loop</h4>
            <p>For each batch, you'll perform the following steps:</p>
            <ol>
              <li>Zero gradients: <code>optimizer.zero_grad()</code></li>
              <li>Forward pass: <code>outputs = model(inputs)</code></li>
              <li>Calculate loss: <code>loss = criterion(outputs, labels)</code></li>
              <li>Backward pass: <code>loss.backward()</code></li>
              <li>Update weights: <code>optimizer.step()</code></li>
              <li>Log/Print progress periodically</li>
            </ol>
          </div>

          <div class="lesson-section">
            <h3>Implementing a Complete Training Pipeline</h3>
            <p>
              Let's implement a complete training pipeline for MNIST digit classification. We'll break it down into sections:
            </p>
            
            <h4>1. Hyperparameters & Setup</h4>
            <p>First, we'll define the key hyperparameters that control our training process:</p>
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3  # Keep it small for a quick test</code></pre>

            <h4>2. Data Loading</h4>
            <p>Next, we'll set up our data pipeline using what we learned in Lesson 3.1:</p>
            <pre><code class="language-python"># Define transformations
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform_pipeline
)

# Create DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)</code></pre>

            <h4>3. Model Definition</h4>
            <p>Now we'll define a simple feed-forward neural network for MNIST classification:</p>
            <pre><code class="language-python"># A simple Feed-Forward Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.relu = nn.ReLU()                          # Activation function
        self.fc2 = nn.Linear(hidden_size, num_classes) # Fully connected layer 2 (output)

    def forward(self, x):
        # Flatten the image first (if it's not already flat)
        # MNIST images are 1x28x28, fc1 expects flat vector
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, 28*28)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # No Softmax here because nn.CrossEntropyLoss includes it
        return out

# Instantiate the model
INPUT_SIZE = 28 * 28  # MNIST images are 28x28 pixels
HIDDEN_SIZE = 128
NUM_CLASSES = 10      # Digits 0-9
model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

print("Model Structure:")
print(model)</code></pre>

            <h4>4. Loss Function and Optimizer</h4>
            <p>We'll choose an appropriate loss function for classification and an optimizer:</p>
            <pre><code class="language-python"># CrossEntropyLoss combines LogSoftmax and NLLLoss
criterion = nn.CrossEntropyLoss()

# Adam optimizer with our specified learning rate
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)</code></pre>

            <h4>5. The Training Loop</h4>
            <p>Finally, we implement the actual training loop that puts everything together:</p>
            <pre><code class="language-python">print("\\nStarting Training...")
model.train()  # Set the model to training mode

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Images shape: [BATCH_SIZE, 1, 28, 28]
        # Labels shape: [BATCH_SIZE]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)  # model's forward method handles flattening

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # Get the scalar value

        if (batch_idx + 1) % 200 == 0:  # Print every 200 mini-batches
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    print(f"--- Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f} ---")

print("Training Finished!")</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Understanding the Training Process</h3>
            <p>
              Let's analyze what happens in this training loop:
            </p>
            <ol>
              <li><strong>Outer loop</strong>: We iterate for a fixed number of epochs (NUM_EPOCHS).</li>
              <li><strong>Inner loop</strong>: For each epoch, we process every batch from the DataLoader.</li>
              <li><strong>Gradient zeroing</strong>: We clear old gradients before each forward pass.</li>
              <li><strong>Forward propagation</strong>: We pass the current batch through our model.</li>
              <li><strong>Loss calculation</strong>: We compare predictions with actual labels.</li>
              <li><strong>Backward propagation</strong>: We compute gradients of the loss with respect to model parameters.</li>
              <li><strong>Parameter update</strong>: We adjust model parameters using the chosen optimizer.</li>
              <li><strong>Progress tracking</strong>: We accumulate and print the loss to monitor training.</li>
            </ol>
            <p>
              As training progresses, you should observe the loss decreasing, which indicates the model is learning to classify digits correctly.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Task 3.2:</strong> Your primary task for this lesson is to understand, implement, and run the training script provided above.
              </p>
              <ol>
                <li><strong>Set up and Run</strong>: Copy the code into your Python environment. Make sure you have PyTorch and Torchvision installed. Run the script. You should see the loss generally decreasing over epochs and batches.</li>
                <li><strong>Observation</strong>:
                  <ul>
                    <li>Observe the shape of images and labels as they come out of the train_loader.</li>
                    <li>Notice how <code>x = x.view(x.size(0), -1)</code> inside the model's forward method is crucial for converting the 2D image batch into a 1D vector batch suitable for <code>nn.Linear</code>.</li>
                    <li>Confirm that the average loss per epoch tends to decrease.</li>
                  </ul>
                </li>
                <li><strong>(Optional Experiment)</strong>: After running it successfully, try one of these modifications:
                  <ul>
                    <li>Change the LEARNING_RATE (e.g., to 0.01 or 0.0001) and see how it affects the training speed and final loss.</li>
                    <li>Change the HIDDEN_SIZE of the neural network (e.g., to 64 or 256).</li>
                    <li>Try using optim.SGD instead of optim.Adam. What changes do you observe in the training process or the loss values?</li>
                  </ul>
                </li>
              </ol>

              <p>
                <strong>Bonus Challenge</strong>: Can you modify the code to add a validation step after each epoch? This would involve:
              </p>
              <ol>
                <li>Loading the MNIST test dataset</li>
                <li>Creating a function to evaluate the model on this dataset</li>
                <li>Calling this function after each training epoch to track validation accuracy</li>
              </ol>
              <p>
                This will give you a better sense of how well your model generalizes to unseen data.
              </p>
            </div>
          </div>
        `,
        completed: false,
      },
      {
        id: 3,
        title: "Evaluation and Validation",
        content: `
          <h2>Evaluation and Validation: Testing Your Model</h2>
          <p>
            Now that we can train a neural network, it's crucial to evaluate its performance on unseen data. In this lesson, we'll learn how to properly evaluate a trained model on a validation or test set.
          </p>

          <div class="lesson-section">
            <h3>The Importance of Separate Evaluation Data</h3>
            <p>
              Training a model and evaluating it on the same data can be misleading. The model might achieve high accuracy on the training data simply by "memorizing" it, without truly learning to generalize. This is called <strong>overfitting</strong>.
            </p>
            <p>
              To assess how well our model will perform on new, unseen data, we need:
            </p>
            <ul>
              <li><strong>Validation set</strong>: Used during development for hyperparameter tuning</li>
              <li><strong>Test set</strong>: Used for final, unbiased evaluation</li>
            </ul>
            <p>
              For MNIST, torchvision conveniently provides a distinct test set that we'll use for evaluation.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Key Evaluation Concepts</h3>
            
            <h4>Evaluation Mode</h4>
            <p>
              Before evaluating your model, you must switch it to evaluation mode by calling <code>model.eval()</code>. This is important because some layers behave differently during training and inference:
            </p>
            <ul>
              <li>Dropout layers are deactivated (no neurons are dropped)</li>
              <li>BatchNorm layers use their learned running mean and variance instead of the current batch statistics</li>
            </ul>
            <p>
              Failing to do this can lead to inconsistent or worse evaluation results.
            </p>
            
            <h4>Disabling Gradient Computation</h4>
            <p>
              During evaluation, we're only performing a forward pass to get predictions; we're not updating the model's weights. Therefore, there's no need to calculate gradients. PyTorch allows you to disable gradient calculations using the <code>with torch.no_grad():</code> context manager. This:
            </p>
            <ul>
              <li>Speeds up computations</li>
              <li>Reduces memory usage</li>
              <li>Prevents accidental weight updates</li>
            </ul>
            
            <h4>Evaluation Metrics</h4>
            <p>
              For classification tasks like MNIST, the most common metric is accuracy:
            </p>
            <pre><code>Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)</code></pre>
            <p>
              Other metrics like precision, recall, and F1-score are also important, especially for imbalanced datasets, but we'll focus on accuracy for this lesson.
            </p>
          </div>

          <div class="lesson-section">
            <h3>The Evaluation Loop Structure</h3>
            <p>
              A typical evaluation loop follows this structure:
            </p>
            <ol>
              <li>Set the model to evaluation mode: <code>model.eval()</code></li>
              <li>Wrap the entire evaluation process in <code>with torch.no_grad():</code></li>
              <li>Initialize counters for correct predictions and total predictions</li>
              <li>Iterate through the DataLoader for your test set</li>
              <li>For each batch:
                <ul>
                  <li>Get the input data and true labels</li>
                  <li>Perform a forward pass: <code>outputs = model(images)</code></li>
                  <li>Get predictions by finding the index of the maximum output value</li>
                  <li>Update counters for correct and total predictions</li>
                </ul>
              </li>
              <li>Calculate the overall accuracy</li>
            </ol>
            <p>
              <strong>Important:</strong> If you're evaluating during a training loop (e.g., after each epoch), remember to switch the model back to training mode using <code>model.train()</code> before resuming training.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Implementing Evaluation for Our MNIST Model</h3>
            
            <h4>1. Loading the Test Dataset</h4>
            <p>First, we need to load the MNIST test dataset:</p>
            <pre><code class="language-python"># Load the MNIST Test Dataset
print("\\nLoading MNIST Test set for evaluation...")
test_transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,  # Get the test set
    download=True,
    transform=test_transform_pipeline
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1000,  # Can use a larger batch size for evaluation
    shuffle=False     # No need to shuffle test data
)
print(f"MNIST Test dataset loaded. Number of test samples: {len(test_dataset)}")</code></pre>

            <h4>2. Evaluation Function</h4>
            <p>Next, let's define a function to evaluate our model:</p>
            <pre><code class="language-python">def evaluate_model(model, test_loader):
    """Evaluate the model on the test set and return accuracy."""
    model.eval()  # Set the model to evaluation mode
    
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient calculations for efficiency
        for images, labels in test_loader:
            # Forward pass
            outputs = model(images)
            
            # Get predictions from the max value in the output logits
            _, predicted = torch.max(outputs.data, 1)
            
            # Update counters
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    # Calculate accuracy
    accuracy = 100 * correct_predictions / total_samples
    return accuracy</code></pre>

            <h4>3. Using the Evaluation Function After Each Epoch</h4>
            <p>Finally, let's modify our training loop to evaluate after each epoch:</p>
            <pre><code class="language-python">print("\\nStarting Training with Evaluation...")

for epoch in range(NUM_EPOCHS):
    # Set model to training mode
    model.train()
    
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # Calculate average training loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"--- Epoch {epoch+1}, Training Loss: {epoch_loss:.4f} ---")
    
    # Evaluate on test set after each epoch
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy after Epoch {epoch+1}: {test_accuracy:.2f}%")
    
    # No need to explicitly set model.train() here as we do it at the beginning of each epoch loop

print("Training and Evaluation Finished!")</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Understanding the Evaluation Process</h3>
            <p>
              Let's break down what's happening during evaluation:
            </p>
            <ol>
              <li><strong>model.eval()</strong>: Switches layers like Dropout and BatchNorm to evaluation behavior.</li>
              <li><strong>torch.no_grad()</strong>: Tells PyTorch not to track tensor operations for backpropagation.</li>
              <li><strong>Forward pass</strong>: We compute the model's predictions for each batch of test images.</li>
              <li><strong>torch.max(outputs.data, 1)</strong>: Finds the index of the maximum value along dimension 1 (the class dimension), which gives us the predicted class.</li>
              <li><strong>(predicted == labels).sum().item()</strong>: Counts how many predictions match the true labels.</li>
              <li><strong>Accuracy calculation</strong>: Divides the number of correct predictions by the total number of samples.</li>
            </ol>
            <p>
              By evaluating after each epoch, we can track not only if our model is learning (decreasing training loss) but also if it's generalizing well to unseen data (increasing test accuracy).
            </p>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Task 3.3:</strong> Integrate the evaluation logic into your training script from Lesson 3.2.
              </p>
              <ol>
                <li><strong>Modify Script from 3.2</strong>: Open your Python script from Lesson 3.2.</li>
                <li><strong>Add Test Data Loading</strong>: Before your main training loop begins, add the code to load the MNIST test dataset and create the test_loader.</li>
                <li><strong>Define the Evaluation Function</strong>: Implement the evaluate_model function as shown above.</li>
                <li><strong>Integrate Evaluation Per Epoch</strong>: Modify your training loop so that after each epoch, the model's performance on the test set is calculated and printed.</li>
                <li><strong>Run and Observe</strong>: Run the complete script. You should now see both the training loss and test accuracy for each epoch.</li>
              </ol>

              <p>
                <strong>Expected Observations</strong>:
              </p>
              <ul>
                <li>The training loss should decrease over epochs</li>
                <li>The test accuracy should increase over epochs</li>
                <li>The test accuracy will likely plateau after a few epochs at around 95-98% for this simple model on MNIST</li>
              </ul>

              <p>
                <strong>Bonus Challenge</strong>: Can you enhance your evaluation with additional metrics or visualizations?
              </p>
              <ul>
                <li>Create a confusion matrix to see which digits are most often confused with each other</li>
                <li>Plot training loss and test accuracy over epochs to visualize learning progress</li>
                <li>Save the model with the best test accuracy during training</li>
              </ul>
            </div>
          </div>

          <div class="lesson-section">
            <h3>Complete Script with Training and Evaluation</h3>
            <p>
              Here's what your complete script might look like with both training and evaluation integrated:
            </p>
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- 1. Hyperparameters & Setup ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3
INPUT_SIZE = 28 * 28  # MNIST images are 28x28 pixels
HIDDEN_SIZE = 128
NUM_CLASSES = 10      # Digits 0-9

# --- 2. Data Loading ---
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Training Dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform_pipeline
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Test Dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform_pipeline
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1000,
    shuffle=False
)

# --- 3. Model Definition ---
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Instantiate the model
model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
print("Model Structure:")
print(model)

# --- 4. Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Evaluation Function ---
def evaluate_model(model, test_loader):
    """Evaluate the model on the test set and return accuracy."""
    model.eval()  # Set the model to evaluation mode
    
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient calculations
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    accuracy = 100 * correct_predictions / total_samples
    return accuracy

# --- 6. Training Loop with Evaluation ---
print("\\nStarting Training with Evaluation...")

for epoch in range(NUM_EPOCHS):
    # Set model to training mode
    model.train()
    
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"--- Epoch {epoch+1}, Training Loss: {epoch_loss:.4f} ---")
    
    # Evaluate on test set
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy after Epoch {epoch+1}: {test_accuracy:.2f}%")

print("Training and Evaluation Finished!")</code></pre>
            <p>
              This script provides a solid foundation for training and evaluating neural networks in PyTorch. As you become more comfortable with these concepts, you can extend it with more advanced techniques like learning rate scheduling, early stopping, and model checkpointing.
            </p>
          </div>
        `,
        completed: false,
      },
      // ...more lessons
    ],
  },
];
