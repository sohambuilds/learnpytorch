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
              <li><strong>Multi-GPU Training:</strong> PyTorch supports distributed training across multiple GPUs using <code>nn.DataParallel</code> or <code>nn.DistributedDataParallel</code> (beyond the scope of this lesson).</li>
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
        title: "Best Practices for a Full Training, Evaluation & Saving Loop",
        content: `
          <h2>Best Practices for a Full Training, Evaluation & Saving Loop</h2>
          <p>
            In this lesson, we'll consolidate our previous learnings into a robust training pipeline that includes training, periodic evaluation, and intelligently saving the best model checkpoints.
          </p>

          <div class="lesson-section">
            <h3>Key Best Practices</h3>
            <p>
              A professional deep learning training pipeline typically incorporates the following practices:
            </p>
            <ul>
              <li><strong>Modular Functions:</strong> Use separate functions for training, evaluation, and other tasks.</li>
              <li><strong>Device Handling:</strong> Consistently use a device variable for model and data.</li>
              <li><strong>Checkpointing Strategy:</strong> Save models based on performance metrics like validation accuracy.</li>
              <li><strong>Resuming Training:</strong> Design your script to be able to load a checkpoint and resume training.</li>
              <li><strong>Clear Logging:</strong> Print informative messages about training progress and results.</li>
              <li><strong>Hyperparameter Organization:</strong> Keep hyperparameters clearly defined at the beginning.</li>
              <li><strong>Consistent Mode Switching:</strong> Use <code>model.train()</code> before training and <code>model.eval()</code> before evaluation.</li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>A Robust Training Loop</h3>
            <p>
              Let's see how to implement a comprehensive training pipeline with these best practices:
            </p>
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# --- 1. Hyperparameters & Setup ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20
INPUT_SIZE = 28 * 28  # MNIST images are 28x28 pixels
HIDDEN_SIZE = 128
NUM_CLASSES = 10      # Digits 0-9
CHECKPOINT_FILENAME = "mnist_best_model_checkpoint.pth"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Data Loading ---
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Training dataset
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

# Test dataset
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

# Instantiate the model and move to device
model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
print("Model Structure:")
print(model)

# --- 4. Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training state initialization ---
START_EPOCH = 0
BEST_VAL_ACCURACY = 0.0

# Optional: Load checkpoint if exists to resume training
if os.path.exists(CHECKPOINT_FILENAME):
    print(f"Loading checkpoint: {CHECKPOINT_FILENAME}")
    # map_location ensures model loads correctly even if current device is different from saving device
    checkpoint = torch.load(CHECKPOINT_FILENAME, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    START_EPOCH = checkpoint['epoch'] # Checkpoint saves the epoch *after* which it was saved
    BEST_VAL_ACCURACY = checkpoint.get('best_val_accuracy', 0.0) # Use .get for backward compatibility
    print(f"Resuming training from epoch {START_EPOCH + 1}. Best validation accuracy so far: {BEST_VAL_ACCURACY:.2f}%")

# --- 6. Training and Evaluation Loop ---
print(f"Starting training from epoch {START_EPOCH + 1} up to {NUM_EPOCHS} on {device}")

for epoch in range(START_EPOCH, NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    train_loss_epoch = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
    avg_train_loss = train_loss_epoch / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Training Loss: {avg_train_loss:.4f}")

    # --- Validation Phase ---
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss_epoch = 0.0
    with torch.no_grad():
        for images, labels in test_loader: # Using test_loader as validation set here
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels) # Can also track validation loss
            val_loss_epoch += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    current_val_accuracy = 100 * val_correct / val_total
    avg_val_loss = val_loss_epoch / len(test_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {current_val_accuracy:.2f}%")

    # Save checkpoint if current model has better validation accuracy
    if current_val_accuracy > BEST_VAL_ACCURACY:
        BEST_VAL_ACCURACY = current_val_accuracy
        print(f"New best validation accuracy: {BEST_VAL_ACCURACY:.2f}%. Saving model checkpoint...")
        checkpoint_data = {
            'epoch': epoch + 1, # Save as the completed epoch number
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': BEST_VAL_ACCURACY,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        torch.save(checkpoint_data, CHECKPOINT_FILENAME)
        print(f"Checkpoint saved to {CHECKPOINT_FILENAME}")
    
    print("-" * 50) # Separator for epochs

print(f"Training finished. Best validation accuracy achieved: {BEST_VAL_ACCURACY:.2f}%")</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Understanding the Enhanced Training Loop</h3>
            <p>
              Let's examine the key components of this improved training pipeline:
            </p>
            <ol>
              <li><strong>Checkpoint Loading:</strong> At the beginning, we check if a checkpoint file exists and load it if found, which allows resuming training from where we left off.</li>
              <li><strong>Training Phase:</strong> We set the model to training mode, process batches, and track the average loss per epoch.</li>
              <li><strong>Validation Phase:</strong> After each training epoch, we evaluate the model on the test set (serving as our validation set here).</li>
              <li><strong>Checkpoint Saving:</strong> We save a checkpoint only when the model achieves a new best validation accuracy, ensuring we keep the best performing model.</li>
              <li><strong>Comprehensive Checkpoint Data:</strong> Our checkpoints include all necessary information to resume training: model state, optimizer state, epoch number, and metrics.</li>
            </ol>
            <p>
              This type of loop provides a robust foundation that you can further enhance with techniques like learning rate scheduling, early stopping, and more detailed logging or visualization.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Task 4.3:</strong> Implement a robust training and checkpointing system.
              </p>
              <ol>
                <li>Refactor your MNIST training script (which should now incorporate GPU support from Task 4.2).</li>
                <li>Implement the checkpointing system:
                  <ul>
                    <li>Initialize a variable <code>best_val_accuracy</code> to <code>0.0</code> before the training loop.</li>
                    <li>After each epoch's validation phase, if the <code>current_val_accuracy</code> is greater than <code>best_val_accuracy</code>:
                      <ul>
                        <li>Update <code>best_val_accuracy</code>.</li>
                        <li>Print a message indicating a new best accuracy and that the model is being saved.</li>
                        <li>Save a checkpoint dictionary containing at least: the current epoch number, <code>model.state_dict()</code>, <code>optimizer.state_dict()</code>, and the <code>best_val_accuracy</code>.</li>
                      </ul>
                    </li>
                  </ul>
                </li>
                <li><strong>Bonus Challenge:</strong> Implement the logic to load this checkpoint at the beginning of your script if the checkpoint file exists. This should allow you to resume training from the saved epoch, using the saved model and optimizer states, and the last known best validation accuracy. Test this by running training for a few epochs, stopping it, and then running it again – it should resume.</li>
              </ol>

              <p>
                This task ties together all the skills you've learned across the modules, creating a professional-quality training pipeline that handles device management, evaluation, and checkpointing.
              </p>
            </div>
          </div>
        `,
        completed: false,
      },
    ],
  },
];
