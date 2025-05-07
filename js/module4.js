window.module4Data = [
  {
    id: 4,
    title: "Practical Considerations",
    description:
      "Learn key practical skills for real-world PyTorch development, including saving models and GPU acceleration.",
    lessons: [
      {
        id: 1,
        title: "Saving and Loading Models",
        content: `
          <h2>Saving and Loading Models</h2>
          <p>
            After training your neural networks, you'll often need to save them for later use. In this lesson, we'll learn how to save and load PyTorch models properly.
          </p>

          <div class="lesson-section">
            <h3>Why Save Models?</h3>
            <p>
              There are several important reasons to save your trained models:
            </p>
            <ul>
              <li><strong>Inference:</strong> Use your trained model to make predictions on new data without retraining.</li>
              <li><strong>Resuming Training:</strong> If training is interrupted or you want to train in stages (e.g., fine-tuning later), you can save your progress and continue.</li>
              <li><strong>Sharing:</strong> Share your trained models with others.</li>
              <li><strong>Checkpointing:</strong> Save intermediate models during a long training process as a backup or to capture the best performing version.</li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>Saving and Loading Methods</h3>
            
            <h4>Method 1: Using model.state_dict() (Recommended)</h4>
            <p>
              The <code>state_dict()</code> is a Python dictionary that maps each layer to its learnable parameters (weights and biases). This approach has several advantages:
            </p>
            <ul>
              <li>It only saves the parameters, not the entire model structure, making it more lightweight and portable.</li>
              <li>It's more flexible and is the recommended approach for most use cases.</li>
            </ul>
            <p>To save a model's state dictionary:</p>
            <pre><code class="language-python">torch.save(model.state_dict(), 'model_weights.pth')</code></pre>
            
            <p>To load the saved state dictionary:</p>
            <pre><code class="language-python"># First, create an instance of your model architecture
model = YourModelClass(*args)

# Then load the saved parameters
model.load_state_dict(torch.load('model_weights.pth'))

# For inference, set the model to evaluation mode
model.eval()</code></pre>
            
            <p>
              <strong>Important:</strong> When loading a state dictionary, ensure the model architecture used for loading matches the one used for saving!
            </p>
            
            <h4>Method 2: Saving the Entire Model (Less Flexible)</h4>
            <p>
              You can also save the entire model object using Python's pickle module:
            </p>
            <pre><code class="language-python">torch.save(model, 'entire_model.pth')</code></pre>
            
            <p>To load the entire model:</p>
            <pre><code class="language-python">model = torch.load('entire_model.pth')
model.eval()  # For inference</code></pre>
            
            <p>
              While simpler, this approach can break if you refactor your code or move the model file to an environment with different class definitions. The <code>state_dict</code> approach is generally more robust.
            </p>
            
            <h4>Method 3: Checkpointing for Resuming Training</h4>
            <p>
              When resuming training, you typically want to save more than just the model's weights. A good checkpoint includes:
            </p>
            <ul>
              <li><code>model.state_dict()</code></li>
              <li><code>optimizer.state_dict()</code> (to resume optimizer's state, like momentum values)</li>
              <li>Current epoch number</li>
              <li>Last recorded training/validation loss</li>
              <li>Any other necessary training state</li>
            </ul>
            <pre><code class="language-python">checkpoint = {
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': current_loss,
    # Add any other info you need
}
torch.save(checkpoint, 'checkpoint.pth')</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Complete Example</h3>
            <p>
              Let's put all these methods together in a comprehensive example:
            </p>
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
import os

# --- Define a simple model (reusing SimpleNN from Module 3) ---
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

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
NUM_CLASSES = 10
LEARNING_RATE = 0.001

model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 1. Saving and Loading model.state_dict() (Recommended) ---
print("--- Method 1: Saving/Loading state_dict ---")
# Imagine the model has been trained...
# For demonstration, let's just print initial weights of fc1
print(f"Initial fc1.weight (excerpt):\\n{model.fc1.weight.data[0, :5]}")

MODEL_WEIGHTS_PATH = 'simple_nn_weights.pth'
torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
print(f"Model state_dict saved to {MODEL_WEIGHTS_PATH}")

# Create a new model instance for loading
new_model_instance = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
# Load the saved state dictionary
new_model_instance.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
print("Loaded state_dict into new model instance.")

# Set to evaluation mode if using for inference
new_model_instance.eval()
print(f"Loaded model fc1.weight (excerpt):\\n{new_model_instance.fc1.weight.data[0, :5]}")
# It should match the original model's weights.

# --- 2. Saving and Loading an entire model (Less flexible) ---
print("\\n--- Method 2: Saving/Loading entire model ---")
ENTIRE_MODEL_PATH = 'entire_simple_nn.pth'
torch.save(model, ENTIRE_MODEL_PATH)
print(f"Entire model saved to {ENTIRE_MODEL_PATH}")

loaded_entire_model = torch.load(ENTIRE_MODEL_PATH)
loaded_entire_model.eval() # Set to eval mode
print("Entire model loaded.")
print(f"Loaded entire model fc1.weight (excerpt):\\n{loaded_entire_model.fc1.weight.data[0, :5]}")


# --- 3. Saving a Checkpoint for Resuming Training ---
print("\\n--- Method 3: Saving a training checkpoint ---")
EPOCH_NUM = 5
current_loss = 0.45 # Example loss
CHECKPOINT_PATH = 'training_checkpoint.pth'

checkpoint = {
    'epoch': EPOCH_NUM,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': current_loss,
    # Add any other info you need
}
torch.save(checkpoint, CHECKPOINT_PATH)
print(f"Training checkpoint saved to {CHECKPOINT_PATH}")

# To load the checkpoint:
model_for_resume = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
optimizer_for_resume = optim.Adam(model_for_resume.parameters(), lr=LEARNING_RATE) # Re-create optimizer

if os.path.exists(CHECKPOINT_PATH):
    loaded_checkpoint = torch.load(CHECKPOINT_PATH)
    model_for_resume.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer_for_resume.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    start_epoch = loaded_checkpoint['epoch']
    previous_loss = loaded_checkpoint['loss']
    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch+1}, last loss: {previous_loss:.4f}")
    model_for_resume.train() # Set to train mode for resuming
else:
    print("No checkpoint found, starting from scratch.")</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Task 4.1:</strong> Save and load your MNIST model for evaluation.
              </p>
              <ol>
                <li>Take your MNIST training script from Lesson 3.3 (the one with training and evaluation).</li>
                <li>After the training loop is complete, save the <code>state_dict()</code> of your trained <code>SimpleNN</code> model to a file (e.g., <code>my_mnist_model.pth</code>).</li>
                <li>Create a <strong>new, separate Python script</strong> (or a new cell in your Jupyter Notebook). In this new script/cell:
                  <ul>
                    <li>Define (or import) the <code>SimpleNN</code> model class exactly as it was during training.</li>
                    <li>Create an instance of <code>SimpleNN</code>.</li>
                    <li>Load the saved <code>state_dict()</code> from <code>my_mnist_model.pth</code> into this new instance.</li>
                    <li>Set the model to evaluation mode (<code>model.eval()</code>).</li>
                    <li>To verify it loaded correctly: Load the MNIST test dataset and <code>test_loader</code> (as you did in 3.3), and calculate the accuracy of this loaded model on the test set. It should be similar to what you observed at the end of training in 3.3.</li>
                  </ul>
                </li>
              </ol>
            </div>
          </div>
        `,
        completed: false,
      },
      {
        id: 2,
        title: "Moving Computations to the GPU",
        content: `
          <h2>Moving Computations to the GPU</h2>
          <p>
            Neural networks can be computationally intensive, especially for large models or datasets. Fortunately, PyTorch makes it easy to leverage GPUs to speed up training and inference.
          </p>

          <div class="lesson-section">
            <h3>Why Use GPUs?</h3>
            <p>
              GPUs (Graphics Processing Units), especially NVIDIA GPUs with CUDA support, can perform the massive parallel computations in deep learning much faster than CPUs. Operations like matrix multiplications, which are at the core of neural networks, are particularly well-suited for GPU acceleration.
            </p>
            <p>
              Training that might take hours or days on a CPU can often be completed in minutes or hours on a modern GPU.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Device-Agnostic PyTorch Code</h3>
            
            <h4>Checking for GPU Availability</h4>
            <p>
              First, we need to check if a CUDA-enabled GPU is available:
            </p>
            <pre><code class="language-python">torch.cuda.is_available()  # Returns True if a CUDA-enabled GPU is detected</code></pre>
            
            <h4>Creating a Device Object</h4>
            <p>
              To write code that works on both CPU and GPU, we'll create a <code>device</code> object:
            </p>
            <pre><code class="language-python">device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")</code></pre>
            
            <h4>Moving the Model to the Device</h4>
            <p>
              Once we have our device, we can move our model to it:
            </p>
            <pre><code class="language-python">model = YourModelClass().to(device)  # This moves all model parameters to the device</code></pre>
            
            <h4>Moving Tensors to the Device</h4>
            <p>
              Any tensors that interact with the model must also be on the same device:
            </p>
            <pre><code class="language-python"># Inside your training/evaluation loop
images = images.to(device)
labels = labels.to(device)

# Now you can use these with your model
outputs = model(images)</code></pre>
            
            <p>
              <strong>Important:</strong> Unlike <code>model.to(device)</code>, the <code>tensor.to(device)</code> operation is <em>not</em> in-place. You must reassign the result.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Complete Training Example with GPU Support</h3>
            <p>
              Let's modify our MNIST training loop to support GPU acceleration:
            </p>
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- 0. Define Model (reusing SimpleNN) ---
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

# --- 1. Hyperparameters & Device Setup ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3 # Keep it small for a quick test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Data Loading (Remains the same, data is moved in the loop) ---
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_pipeline)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. Model Instantiation and Move to Device ---
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
NUM_CLASSES = 10
model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device) # Move model to device!

# --- 4. Loss Function and Optimizer (Optimizer handles params already on device) ---
criterion = nn.CrossEntropyLoss() # Loss function itself doesn't need to be moved explicitly
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training Loop (Modified to move data to device) ---
print("\\nStarting Training on", device, "...")
model.train()

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    epoch_loss = running_loss / len(train_loader)
    print(f"--- Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f} ---")

print("Training Finished!")</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Key Points About GPU Usage</h3>
            <ul>
              <li><strong>Consistency is Key:</strong> All tensors and the model involved in a single operation must reside on the same device.</li>
              <li><strong>Memory Management:</strong> GPUs have limited memory. If you encounter "CUDA out of memory" errors, try reducing batch size or model size.</li>
              <li><strong>Multi-GPU Training:</strong> PyTorch supports distributed training across multiple GPUs using <code>nn.DataParallel</code> or <code>nn.DistributedDataParallel</code> (covered in the next lesson).</li>
              <li><strong>Clear Unused Variables:</strong> To free up GPU memory, you can delete tensors using <code>del tensor_name</code> or move them to CPU with <code>tensor.cpu()</code>.</li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Task 4.2:</strong> Adapt your MNIST script for GPU acceleration.
              </p>
              <ol>
                <li>Take your complete MNIST training and evaluation script from Task 3.3.</li>
                <li>Modify it to make it device-agnostic:
                  <ul>
                    <li>Define the <code>device</code> object at the beginning.</li>
                    <li>Move your <code>model</code> to this <code>device</code> after instantiation.</li>
                    <li>Inside your training loop, move the <code>images</code> and <code>labels</code> tensors to the <code>device</code> for each batch.</li>
                    <li>Inside your evaluation loop (for the test set), also move the <code>images</code> and <code>labels</code> tensors to the <code>device</code> for each batch.</li>
                  </ul>
                </li>
                <li>Run the script.
                  <ul>
                    <li>If you have a CUDA-enabled GPU and PyTorch with CUDA support installed, it should run on the GPU, and you might notice it's faster, especially with more epochs or a larger model.</li>
                    <li>If you don't have a GPU, it should gracefully fall back to using the CPU and run correctly.</li>
                  </ul>
                </li>
              </ol>
            </div>
          </div>
        `,
        completed: false,
      },
      {
        id: 3,
        title: "Introduction to Multi-GPU Training with DataParallel",
        content: `
          <h2>Introduction to Multi-GPU Training with DataParallel</h2>
          <p>
            As deep learning models grow larger and datasets become more extensive, training on a single GPU can be time-consuming. In this lesson, we'll learn how to leverage multiple GPUs to speed up training through data parallelism.
          </p>

          <div class="lesson-section">
            <h3>Why Multi-GPU?</h3>
            <p>
              Deep learning models are becoming larger and datasets more extensive. Training on a single GPU can be time-consuming. Using multiple GPUs can significantly speed up the training process by distributing the workload, allowing you to:
            </p>
            <ul>
              <li>Train larger models</li>
              <li>Use larger batch sizes</li>
              <li>Complete training in less time</li>
              <li>Experiment more quickly</li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>Understanding Data Parallelism</h3>
            <p>
              Data parallelism is the most common form of parallelism for training deep learning models. The core idea is:
            </p>
            <ol>
              <li>Replicate the model on each available GPU</li>
              <li>Split a mini-batch of data into smaller sub-batches (shards)</li>
              <li>Send each sub-batch to a different GPU</li>
              <li>Each GPU performs the forward pass with its model replica and its shard of data</li>
              <li>Gradients are computed on each GPU</li>
              <li>Gradients from all GPUs are typically summed up on a primary GPU (often GPU 0)</li>
              <li>The model parameters on the primary GPU are updated</li>
              <li>The updated parameters are then broadcast back to all other GPU replicas</li>
            </ol>
            <p>
              This approach allows you to process more data in parallel without changing your model architecture.
            </p>
          </div>

          <div class="lesson-section">
            <h3>PyTorch's DataParallel</h3>
            <p>
              PyTorch provides <code>torch.nn.DataParallel</code>, a module that makes implementing data parallelism straightforward:
            </p>
            <ul>
              <li>You wrap your existing model with DataParallel: <code>model = nn.DataParallel(model)</code></li>
              <li>DataParallel handles the model replication, data scattering, gradient gathering, and parameter broadcasting automatically</li>
              <li>Your batch size should ideally be a multiple of the number of GPUs for optimal load balancing</li>
            </ul>
            <p>
              <strong>How it Works (Simplified):</strong>
            </p>
            <p>
              When you call <code>model(input_batch)</code> on a DataParallel-wrapped model:
            </p>
            <ul>
              <li>The <code>input_batch</code> is split along the batch dimension (dimension 0)</li>
              <li>Each chunk is sent to a different GPU</li>
              <li>The model is replicated on each GPU</li>
              <li>Forward pass happens in parallel</li>
              <li>Outputs are gathered back to the primary GPU (GPU 0 by default)</li>
              <li>During <code>loss.backward()</code>, gradients are computed on each GPU and then summed on the primary GPU</li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>Considerations and Limitations</h3>
            <ul>
              <li><strong>Single Machine:</strong> DataParallel is designed for multiple GPUs on a single machine. For multi-machine training, <code>torch.nn.parallel.DistributedDataParallel</code> (DDP) is preferred.</li>
              <li><strong>GIL and Master GPU Bottleneck:</strong> Python's Global Interpreter Lock (GIL) can sometimes be a bottleneck. Also, the primary GPU (GPU 0) does more work (gathering outputs, summing gradients, parameter updates), which can lead to uneven GPU utilization.</li>
              <li><strong>Model Saving:</strong> When using DataParallel, the model is wrapped inside <code>model.module</code>. So, when saving the state_dict, you should save <code>model.module.state_dict()</code> to get the original model's parameters, not the DataParallel wrapper's state.</li>
              <li><strong>Graceful Fallback:</strong> It's crucial to write code that checks for the number of available GPUs and only uses DataParallel if more than one is present. Otherwise, it should run on a single GPU or CPU.</li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>Code Example: MNIST Training with DataParallel</h3>
            <p>
              The following code demonstrates how to modify a typical training script to use DataParallel. It will use all available CUDA GPUs if more than one is found; otherwise, it will run on a single GPU or CPU.
            </p>
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os # For saving model

# --- 0. Define Model (reusing SimpleNN) ---
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the image
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- 1. Hyperparameters & Device Setup ---
LEARNING_RATE = 0.001
# For DataParallel, effective batch size is BATCH_SIZE.
# It will be split across GPUs.
# Ensure BATCH_SIZE is >= number of GPUs.
BATCH_SIZE = 128 # Increased batch size for multi-GPU
NUM_EPOCHS = 3
MODEL_SAVE_PATH = "mnist_dp_model.pth"

# Determine device and number of GPUs
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available. Number of GPUs: {num_gpus}")
    if num_gpus > 0:
        device = torch.device("cuda") # Default to cuda:0 if multiple, DataParallel handles distribution
    else: # Should not happen if torch.cuda.is_available() is true and count is 0, but good practice
        print("CUDA available but no GPUs found, using CPU.")
        device = torch.device("cpu")
        num_gpus = 0 # Ensure num_gpus is 0
else:
    print("CUDA not available. Using CPU.")
    device = torch.device("cpu")
    num_gpus = 0

# --- 2. Data Loading ---
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_pipeline)
# DataLoader batch_size is the total batch size that will be split
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 if num_gpus > 0 else 0)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_pipeline)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False, num_workers=4 if num_gpus > 0 else 0)


# --- 3. Model Instantiation ---
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
NUM_CLASSES = 10

# Instantiate the model
model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

# --- Apply DataParallel if multiple GPUs are available ---
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs via nn.DataParallel.")
    model = nn.DataParallel(model) # Wrap the model

# Move model to the primary device (CPU or GPU0)
# DataParallel will handle distributing to other GPUs from GPU0
model.to(device)
print(f"Model moved to: {device}")


# --- 4. Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
# Optimizer receives parameters from the model (or model.module if DataParallel was used)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training Loop ---
print(f"\\nStarting Training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to the primary device. DataParallel will scatter it.
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels)
        loss.backward() # Backward pass
        optimizer.step() # Update weights

        running_loss += loss.item()
        if (batch_idx + 1) % 200 == 0: # Print every 200 mini-batches
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_epoch_loss = running_loss / len(train_loader)
    print(f"--- Epoch {epoch+1} finished. Average Training Loss: {avg_epoch_loss:.4f} ---")

    # --- Evaluation Phase (after each epoch) ---
    model.eval() # Set model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad(): # Disable gradient calculations
        for images_test, labels_test in test_loader:
            images_test = images_test.to(device)
            labels_test = labels_test.to(device)

            outputs_test = model(images_test)
            _, predicted = torch.max(outputs_test.data, 1)

            total_samples += labels_test.size(0)
            correct_predictions += (predicted == labels_test).sum().item()

    accuracy = 100 * correct_predictions / total_samples
    print(f"Validation Accuracy after Epoch {epoch+1}: {accuracy:.2f}%")
    print("-" * 50)

print("Training Finished!")

# --- 6. Saving the model ---
# If DataParallel was used, save model.module.state_dict()
# Otherwise, save model.state_dict()
print("Saving model...")
if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
    print(f"Model (from model.module.state_dict()) saved to {MODEL_SAVE_PATH}")
else:
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model (from model.state_dict()) saved to {MODEL_SAVE_PATH}")

# To load:
# loaded_model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
# loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# loaded_model.to(device) # Move to device
# loaded_model.eval()
# ... then use for inference</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Key Points to Note in the Code</h3>
            <ol>
              <li>We determine the number of available GPUs using <code>torch.cuda.device_count()</code></li>
              <li>We only wrap the model with <code>nn.DataParallel</code> if multiple GPUs are available</li>
              <li>The batch size should be larger than the number of GPUs for optimal utilization</li>
              <li>When saving the model, we check if it's wrapped with DataParallel and save <code>model.module.state_dict()</code> if it is</li>
              <li>The code gracefully falls back to single-GPU or CPU operation if multiple GPUs aren't available</li>
            </ol>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Task 4.3:</strong> Explore multi-GPU training with DataParallel.
              </p>
              
              <h4>Part 1: Understand the Code</h4>
              <p>Carefully read through the provided code example. Pay attention to:</p>
              <ul>
                <li>How <code>torch.cuda.device_count()</code> is used</li>
                <li>How <code>nn.DataParallel(model)</code> wraps the original model</li>
                <li>How data (images, labels) is moved to the device in the training and evaluation loops</li>
                <li>How the model state is saved correctly using <code>model.module.state_dict()</code> if DataParallel was used</li>
              </ul>
              
              <h4>Part 2: Run the Script</h4>
              <p>Execute the script and observe its behavior based on your hardware:</p>
              <ul>
                <li><strong>If you have multiple CUDA GPUs:</strong> Observe if the script utilizes them (you might need to use tools like <code>nvidia-smi</code> in your terminal to monitor GPU usage). You should see a message like "Using X GPUs via nn.DataParallel."</li>
                <li><strong>If you have one CUDA GPU:</strong> The script should run on that single GPU without using DataParallel. You'll see "CUDA is available. Number of GPUs: 1".</li>
                <li><strong>If you only have a CPU:</strong> The script should run on the CPU. You'll see "CUDA not available. Using CPU."</li>
              </ul>
              
              <h4>Part 3: Conceptual Questions</h4>
              <p>Think about these questions (especially if you don't have a multi-GPU setup):</p>
              <ol>
                <li>What would happen if your BATCH_SIZE was very small (e.g., smaller than the number of GPUs)? How might this affect efficiency?</li>
                <li>Why is <code>model.module.state_dict()</code> important for saving when using DataParallel?</li>
                <li>How would you modify this code to use a specific subset of available GPUs instead of all of them?</li>
              </ol>
              
              <p>
                <strong>Note:</strong> Even if you don't have multiple GPUs, the primary goal of this task is to understand the code structure and concepts. The script is designed to run correctly on a single GPU or CPU, with the multi-GPU specific parts (nn.DataParallel wrapping) being skipped if num_gpus ≤ 1.
              </p>
              
              <p>
                For more advanced and often more efficient multi-GPU or distributed training, <code>torch.nn.parallel.DistributedDataParallel</code> is the recommended tool, but it has a slightly steeper learning curve. DataParallel is a good starting point for single-node, multi-GPU scenarios.
              </p>
            </div>
          </div>
        `,
        completed: false,
      },
  
    ],
  },
];
