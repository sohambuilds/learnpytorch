window.module5Data = [
  {
    id: 5,
    title: "Advanced Computer Vision with PyTorch",
    description:
      "Learn to build and train convolutional neural networks (CNNs) for image classification tasks beyond MNIST.",
    lessons: [
      {
        id: 1,
        title: "Building Your First CNN for CIFAR-10",
        content: `
          <h2>Building Your First CNN for CIFAR-10</h2>
          <p>
            This lesson will introduce you to Convolutional Neural Networks (CNNs) by building one from scratch for the CIFAR-10 dataset.
          </p>

          <div class="lesson-section">
            <h3>Objective</h3>
            <p>
              Understand the fundamental components of Convolutional Neural Networks (CNNs) by building one from scratch for the CIFAR-10 dataset, and train it using the skills acquired in previous modules.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Recall: Why CNNs for Images?</h3>
            <p>
              In Module 3, we used a simple Fully Connected Network (FCN) for MNIST. While it worked, FCNs struggle with more complex image data like CIFAR-10 (32x32 color images, 10 classes: airplane, car, bird, etc.) because:
            </p>
            <ol>
              <li><strong>Loss of Spatial Information:</strong> Flattening images discards the 2D structure.</li>
              <li><strong>Parameter Inefficiency:</strong> FCNs require too many parameters for images, leading to overfitting and high computational cost.</li>
              <li><strong>Not Translation Invariant:</strong> An object recognized in one location might not be recognized elsewhere.</li>
            </ol>
            <p>
              CNNs are designed to overcome these by using specialized layers that preserve spatial structure and efficiently learn hierarchical features.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Building Blocks: <code>nn.Module</code> and New CNN Layers</h3>
            <p>
              Just like our FCN, our CNN will be a class inheriting from <code>torch.nn.Module</code>. We'll define its layers in <code>__init__</code> and the data flow in <code>forward</code>. The new key layers for CNNs are:
            </p>

            <h4>1. <code>nn.Conv2d</code> (Convolutional Layer)</h4>
            <p>
              The workhorse of CNNs. It applies a set of learnable <strong>filters</strong> (or kernels) across the input image (or feature map from a previous layer). Each filter is a small grid of weights designed to detect specific local patterns (edges, textures, corners, etc.).
            </p>
            <p><strong>Key Parameters we'll use:</strong></p>
            <ul>
              <li><code>in_channels</code>: Number of channels in the input. For the first layer processing CIFAR-10 (RGB images), <code>in_channels=3</code>. For subsequent layers, it's the <code>out_channels</code> of the <em>previous</em> <code>Conv2d</code> layer.</li>
              <li><code>out_channels</code>: Number of filters to apply. Each filter produces one output feature map. This determines the "depth" of the output.</li>
              <li><code>kernel_size</code>: The dimensions of the filter (e.g., <code>3</code> for a 3x3 filter).</li>
              <li><code>stride</code>: How many pixels the filter slides at a time (default is 1).</li>
              <li><code>padding</code>: Zero-padding added to the input's borders. This helps control the output size. <code>padding=1</code> with a <code>kernel_size=3</code> and <code>stride=1</code> often keeps the height and width the same.</li>
            </ul>

            <pre><code class="language-python"># Example of a convolutional layer
import torch.nn as nn

# First conv layer for RGB images (3 channels)
conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
                  stride=1, padding=1)
# This creates 16 filters of size 3x3, applied with stride 1
# padding=1 ensures output size matches input size
</code></pre>

            <h4>2. <code>nn.ReLU</code> (Activation Function)</h4>
            <p>
              Same as before! Applied after each convolutional layer to introduce non-linearity.
            </p>

            <pre><code class="language-python"># Applying ReLU activation after convolution
relu = nn.ReLU()
# Used as: output = relu(conv_output)
</code></pre>

            <h4>3. <code>nn.MaxPool2d</code> (Max Pooling Layer)</h4>
            <p>
              Reduces the spatial dimensions (height and width) of the feature maps, making the network more computationally efficient and providing some robustness to small translations.
            </p>
            <p>
              It works by taking the maximum value from a small window (defined by <code>kernel_size</code>).
            </p>
            <p><strong>Key Parameters we'll use:</strong></p>
            <ul>
              <li><code>kernel_size</code>: The size of the window (e.g., <code>2</code> for a 2x2 window).</li>
              <li><code>stride</code>: How many pixels the window slides. Often set equal to <code>kernel_size</code> (e.g., <code>stride=2</code>) to halve the dimensions.</li>
            </ul>

            <pre><code class="language-python"># Max pooling layer
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# This reduces the spatial dimensions by half
# If input is 32x32, output will be 16x16
</code></pre>

            <h4>4. <code>nn.Linear</code> (Fully Connected Layer)</h4>
            <p>
              Used at the <em>end</em> of the CNN, after features are extracted and "flattened" into a 1D vector, to perform the final classification.
            </p>

            <pre><code class="language-python"># Fully connected layer for classification
# Assuming we have flattened our features to size 1024
fc = nn.Linear(1024, 10)  # 10 classes for CIFAR-10
</code></pre>

            <h4>5. <code>nn.Dropout</code> (Optional but good for regularization)</h4>
            <p>
              Randomly zeros out some neurons during training to reduce overfitting.
            </p>

            <pre><code class="language-python"># Dropout for regularization
dropout = nn.Dropout(0.5)  # 50% dropout rate
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Designing Our CIFAR-10 CNN Architecture (Step-by-Step)</h3>
            <p>
              Let's design a simple CNN. CIFAR-10 images are <code>3x32x32</code> (Channels x Height x Width).
            </p>

            <h4>1. Convolutional Block 1:</h4>
            <ul>
              <li><code>self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)</code> -> Output: <code>(Batch Size, 16, 32, 32)</code></li>
              <li><code>self.relu1 = nn.ReLU()</code></li>
              <li><code>self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)</code> -> Output: <code>(Batch Size, 16, 16, 16)</code></li>
            </ul>

            <pre><code class="language-python"># Convolutional Block 1
self.conv_block1 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # Out: Bx16x32x32
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2) # Out: Bx16x16x16
)
</code></pre>

            <h4>2. Convolutional Block 2:</h4>
            <ul>
              <li><code>self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)</code> -> Output: <code>(Batch Size, 32, 16, 16)</code></li>
              <li><code>self.relu2 = nn.ReLU()</code></li>
              <li><code>self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)</code> -> Output: <code>(Batch Size, 32, 8, 8)</code></li>
            </ul>

            <pre><code class="language-python"># Convolutional Block 2
self.conv_block2 = nn.Sequential(
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # Out: Bx32x16x16
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2) # Out: Bx32x8x8
)
</code></pre>

            <h4>3. Convolutional Block 3:</h4>
            <ul>
              <li><code>self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)</code> -> Output: <code>(Batch Size, 64, 8, 8)</code></li>
              <li><code>self.relu3 = nn.ReLU()</code></li>
              <li><code>self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)</code> -> Output: <code>(Batch Size, 64, 4, 4)</code></li>
            </ul>

            <pre><code class="language-python"># Convolutional Block 3
self.conv_block3 = nn.Sequential(
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # Out: Bx64x8x8
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2) # Out: Bx64x4x4
)
</code></pre>

            <h4>4. Flattening:</h4>
            <ul>
              <li>The output <code>(Batch Size, 64, 4, 4)</code> is flattened to <code>(Batch Size, 64 * 4 * 4)</code> which is <code>(Batch Size, 1024)</code>.</li>
              <li>Done via <code>x = x.view(x.size(0), -1)</code>.</li>
            </ul>

            <pre><code class="language-python"># In the forward method
x = x.view(x.size(0), -1) # Flatten: Bx1024
</code></pre>

            <h4>5. Fully Connected (FC) Layers for Classification:</h4>
            <ul>
              <li><code>self.fc1 = nn.Linear(1024, 256)</code></li>
              <li><code>self.relu_fc1 = nn.ReLU()</code></li>
              <li><code>self.dropout = nn.Dropout(0.5)</code></li>
              <li><code>self.fc2 = nn.Linear(256, num_classes)</code> (where <code>num_classes=10</code> for CIFAR-10)</li>
            </ul>

            <pre><code class="language-python"># Classifier
self.flattened_features = 64 * 4 * 4 # 1024
self.classifier = nn.Sequential(
    nn.Linear(self.flattened_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
</code></pre>

            <h4>Putting It All Together: The Complete CNN Model</h4>
            <pre><code class="language-python"># The complete SimpleCNN model definition
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # Out: Bx16x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Out: Bx16x16x16
        )
        # Convolutional Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # Out: Bx32x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Out: Bx32x8x8
        )
        # Convolutional Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # Out: Bx64x8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Out: Bx64x4x4
        )
        # Classifier
        self.flattened_features = 64 * 4 * 4 # 1024
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1) # Flatten: Bx1024
        x = self.classifier(x)
        return x
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Data Preparation for CIFAR-10</h3>
            <p>
              For CIFAR-10, we need to prepare our data differently than we did for MNIST. Let's set up the data loading and transformations:
            </p>

            <pre><code class="language-python">import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-10 specific mean and standard deviation for normalization
cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]

# Training data transformations (with data augmentation)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Random crop with padding
    transforms.RandomHorizontalFlip(p=0.5), # Flip images horizontally with 50% probability
    transforms.ToTensor(),                  # Convert to tensor
    transforms.Normalize(cifar10_mean, cifar10_std) # Normalize
])

# Test data transformations (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

# Loading the datasets
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=100, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=100, shuffle=False, num_workers=2)

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Training Loop and Device Setup</h3>
            <p>
              Let's set up our training loop, building on what we learned in previous modules. First, we'll set up our device and hyperparameters:
            </p>

            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
import os

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 100
NUM_EPOCHS = 15
MODEL_SAVE_PATH = "cifar10_cnn_final_model.pth"
CHECKPOINT_FILENAME = "cifar10_cnn_best_checkpoint.pth"

# Determine device (CPU or GPU)
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available. Number of GPUs: {num_gpus}")
    device = torch.device("cuda")
else:
    print("CUDA not available. Using CPU.")
    device = torch.device("cpu")
    num_gpus = 0
</code></pre>
            
            <p>Now we'll instantiate our model, define loss function and optimizer:</p>

            <pre><code class="language-python"># Create model instance
model = SimpleCNN(num_classes=len(classes))

# Use DataParallel if multiple GPUs are available
if num_gpus > 1 and torch.cuda.is_available():
    print(f"Using {num_gpus} GPUs via nn.DataParallel.")
    model = nn.DataParallel(model)

# Move model to device (CPU or GPU)
model.to(device)

# Print model structure
print("\nCNN Model Structure:")
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params:,}")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
</code></pre>

            <p>Finally, let's implement the training and evaluation loop with checkpointing:</p>

            <pre><code class="language-python"># Training and evaluation variables
START_EPOCH = 0
BEST_VAL_ACCURACY = 0.0

# Resume from checkpoint if available
if os.path.exists(CHECKPOINT_FILENAME):
    print(f"\nLoading checkpoint: {CHECKPOINT_FILENAME}")
    checkpoint = torch.load(CHECKPOINT_FILENAME, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    START_EPOCH = checkpoint.get('epoch', 0)
    BEST_VAL_ACCURACY = checkpoint.get('best_val_accuracy', 0.0)
    print(f"Resuming training from epoch {START_EPOCH + 1}. Best val accuracy: {BEST_VAL_ACCURACY:.2f}%")

print(f"\nStarting training from epoch {START_EPOCH + 1} up to {NUM_EPOCHS} on {device}...")

# Main training loop
for epoch in range(START_EPOCH, NUM_EPOCHS):
    # Training phase
    model.train()
    running_train_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"E[{epoch+1}/{NUM_EPOCHS}],S[{batch_idx+1}/{len(train_loader)}],Loss:{loss.item():.4f}")
    avg_train_loss = running_train_loss / len(train_loader)
    print(f"--- E[{epoch+1}] Avg Train Loss: {avg_train_loss:.4f} ---")
    
    # Evaluation phase
    model.eval()
    running_val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images_val, labels_val in test_loader:
            images_val, labels_val = images_val.to(device), labels_val.to(device)
            outputs_val = model(images_val)
            loss_val = criterion(outputs_val, labels_val)
            running_val_loss += loss_val.item()
            _, predicted_classes = torch.max(outputs_val.data, 1)
            val_total += labels_val.size(0)
            val_correct += (predicted_classes == labels_val).sum().item()
    avg_val_loss = running_val_loss / len(test_loader)
    current_val_accuracy = 100 * val_correct / val_total
    print(f"E[{epoch+1}] Val Loss: {avg_val_loss:.4f}, Val Acc: {current_val_accuracy:.2f}%")
    
    # Save checkpoint if we have a new best validation accuracy
    if current_val_accuracy > BEST_VAL_ACCURACY:
        BEST_VAL_ACCURACY = current_val_accuracy
        print(f"New best val acc: {BEST_VAL_ACCURACY:.2f}%. Saving checkpoint...")
        state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        checkpoint_data = {
            'epoch': epoch + 1, 
            'model_state_dict': state_dict_to_save,
            'optimizer_state_dict': optimizer.state_dict(), 
            'best_val_accuracy': BEST_VAL_ACCURACY,
            'train_loss': avg_train_loss, 
            'val_loss': avg_val_loss
        }
        torch.save(checkpoint_data, CHECKPOINT_FILENAME)
    print("-" * 70)

# Training finished
print(f"Training finished. Best val acc: {BEST_VAL_ACCURACY:.2f}%")
final_model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
torch.save(final_model_state_dict, MODEL_SAVE_PATH)
print(f"Final model state saved to {MODEL_SAVE_PATH}")
print(f"To use best model, load '{CHECKPOINT_FILENAME}'.")
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Complete Code Example</h3>
            <p>Here's the complete code for reference:</p>
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# --- 1. Hyperparameters & Device Setup ---
LEARNING_RATE = 0.001
BATCH_SIZE = 100
NUM_EPOCHS = 15
MODEL_SAVE_PATH = "cifar10_cnn_final_model.pth"
CHECKPOINT_FILENAME = "cifar10_cnn_best_checkpoint.pth"

# Determine device
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available. Number of GPUs: {num_gpus}")
    device = torch.device("cuda")
else:
    print("CUDA not available. Using CPU.")
    device = torch.device("cpu")
    num_gpus = 0 # Ensure num_gpus is defined for later logic if needed

# --- 2. Data Loading & Transformation for CIFAR-10 ---
cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

print("Loading CIFAR-10 dataset...")
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2 if num_gpus > 0 else 0)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2 if num_gpus > 0 else 0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
NUM_CLASSES = len(classes)
print(f"CIFAR-10 dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# --- 3. CNN Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        # Convolutional Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # Out: Bx16x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Out: Bx16x16x16
        )
        # Convolutional Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # Out: Bx32x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Out: Bx32x8x8
        )
        # Convolutional Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # Out: Bx64x8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Out: Bx64x4x4
        )
        # Classifier
        self.flattened_features = 64 * 4 * 4 # 1024
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1) # Flatten: Bx1024
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=NUM_CLASSES)

if num_gpus > 1 and torch.cuda.is_available(): # Check again for safety
    print(f"Using {num_gpus} GPUs via nn.DataParallel.")
    model = nn.DataParallel(model)

model.to(device)
print("\nCNN Model Structure:")
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params:,}")

# --- 4. Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training & Evaluation Loop (with Checkpointing) ---
START_EPOCH = 0
BEST_VAL_ACCURACY = 0.0

if os.path.exists(CHECKPOINT_FILENAME):
    print(f"\nLoading checkpoint: {CHECKPOINT_FILENAME}")
    checkpoint = torch.load(CHECKPOINT_FILENAME, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    START_EPOCH = checkpoint.get('epoch', 0)
    BEST_VAL_ACCURACY = checkpoint.get('best_val_accuracy', 0.0)
    print(f"Resuming training from epoch {START_EPOCH + 1}. Best val accuracy: {BEST_VAL_ACCURACY:.2f}%")

print(f"\nStarting training from epoch {START_EPOCH + 1} up to {NUM_EPOCHS} on {device}...")
for epoch in range(START_EPOCH, NUM_EPOCHS):
    model.train()
    running_train_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"E[{epoch+1}/{NUM_EPOCHS}],S[{batch_idx+1}/{len(train_loader)}],Loss:{loss.item():.4f}")
    avg_train_loss = running_train_loss / len(train_loader)
    print(f"--- E[{epoch+1}] Avg Train Loss: {avg_train_loss:.4f} ---")

    model.eval()
    running_val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images_val, labels_val in test_loader:
            images_val, labels_val = images_val.to(device), labels_val.to(device)
            outputs_val = model(images_val)
            loss_val = criterion(outputs_val, labels_val)
            running_val_loss += loss_val.item()
            _, predicted_classes = torch.max(outputs_val.data, 1)
            val_total += labels_val.size(0)
            val_correct += (predicted_classes == labels_val).sum().item()
    avg_val_loss = running_val_loss / len(test_loader)
    current_val_accuracy = 100 * val_correct / val_total
    print(f"E[{epoch+1}] Val Loss: {avg_val_loss:.4f}, Val Acc: {current_val_accuracy:.2f}%")

    if current_val_accuracy > BEST_VAL_ACCURACY:
        BEST_VAL_ACCURACY = current_val_accuracy
        print(f"New best val acc: {BEST_VAL_ACCURACY:.2f}%. Saving checkpoint...")
        state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        checkpoint_data = {
            'epoch': epoch + 1, 'model_state_dict': state_dict_to_save,
            'optimizer_state_dict': optimizer.state_dict(), 'best_val_accuracy': BEST_VAL_ACCURACY,
            'train_loss': avg_train_loss, 'val_loss': avg_val_loss
        }
        torch.save(checkpoint_data, CHECKPOINT_FILENAME)
    print("-" * 70)

print(f"Training finished. Best val acc: {BEST_VAL_ACCURACY:.2f}%")
final_model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
torch.save(final_model_state_dict, MODEL_SAVE_PATH)
print(f"Final model state saved to {MODEL_SAVE_PATH}")
print(f"To use best model, load '{CHECKPOINT_FILENAME}'.")
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Task 5.1:</strong>
              </p>
              <ol>
                <li><strong>Study the Code:</strong>
                  <ul>
                    <li>Trace data dimensions through the <code>SimpleCNN</code> model.</li>
                    <li>Understand <code>nn.Conv2d</code>, <code>nn.MaxPool2d</code> parameters.</li>
                    <li>Note the use of <code>nn.Sequential</code> for grouping layers.</li>
                    <li>Understand data augmentation (<code>RandomCrop</code>, <code>RandomHorizontalFlip</code>) for training data.</li>
                  </ul>
                </li>
                <li><strong>Run the Script:</strong> Execute the code. Training will take longer than MNIST.</li>
                <li><strong>Observe and Analyze Performance:</strong>
                  <ul>
                    <li>Monitor training loss and validation accuracy.</li>
                    <li>A simple CNN might achieve 60-75%+ validation accuracy on CIFAR-10 after 10-20 epochs.</li>
                  </ul>
                </li>
                <li><strong>(Optional Experimentation):</strong>
                  <ul>
                    <li>Change <code>LEARNING_RATE</code>, <code>BATCH_SIZE</code>, <code>NUM_EPOCHS</code>.</li>
                    <li>Modify CNN architecture: <code>out_channels</code> in <code>Conv2d</code>, neurons in FC layers, remove <code>Dropout</code>.</li>
                    <li>Observe impacts on trainable parameters and performance.</li>
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
        title: "Transfer Learning for Image Classification",
        content: `
          <h2>Transfer Learning for Image Classification</h2>
          <p>
            In this lesson, we'll leverage pre-trained deep Convolutional Neural Networks (CNNs) to achieve strong performance on CIFAR-10, especially when you have limited data for your specific task.
          </p>

          <div class="lesson-section">
            <h3>Objective</h3>
            <p>
              Learn how to use pre-trained deep Convolutional Neural Networks (CNNs) to achieve strong performance on new image classification tasks, especially when you have limited data for your specific task.
            </p>
          </div>

          <div class="lesson-section">
            <h3>What is Transfer Learning?</h3>
            <p>
              Imagine trying to teach a child to recognize a specific breed of dog. It's much easier if the child already understands what a "dog" is, what "fur" is, what "legs" are, etc. Transfer learning in deep learning is similar.
            </p>
            <h4>Leveraging Prior Knowledge</h4>
            <p>
              Models trained on very large and diverse datasets (like ImageNet, which has millions of images across 1000 different everyday object categories) learn a rich hierarchy of visual features. Early layers might learn to detect edges and simple textures, middle layers might recognize more complex patterns like object parts (wheels, eyes, leaves), and later layers learn to identify whole objects.
            </p>
            <h4>Generalizability</h4>
            <p>
              These learned features are often general enough to be useful for other computer vision tasks, even if your specific dataset is different (e.g., classifying types of flowers, medical images, or our CIFAR-10 dataset).
            </p>
            <h4>Benefits</h4>
            <ul>
              <li><strong>Reduced Training Time:</strong> You don't need to train a very deep network from random initialization, which can take a long time.</li>
              <li><strong>Lower Data Requirement:</strong> You can often achieve good results with a smaller dataset for your specific task because the model has already learned general features.</li>
              <li><strong>Better Performance:</strong> Pre-trained models often provide a better starting point, leading to higher accuracy on your target task.</li>
            </ul>
          </div>

          <div class="lesson-section">
            <h3>Using Pre-trained Models from torchvision.models</h3>
            <p>
              PyTorch, through its <code>torchvision.models</code> module, provides easy access to many famous CNN architectures with weights pre-trained on ImageNet. Some popular ones include:
            </p>
            <ul>
              <li>ResNet (e.g., ResNet18, ResNet34, ResNet50)</li>
              <li>VGG (e.g., VGG16, VGG19)</li>
              <li>AlexNet</li>
              <li>GoogLeNet</li>
              <li>MobileNet</li>
              <li>DenseNet</li>
            </ul>
            <p>
              To load a pre-trained model, you can do something like this (using ResNet18 as an example, which is a good balance of size and performance for learning):
            </p>
            <pre><code class="language-python">from torchvision.models import resnet18, ResNet18_Weights
# Load ResNet18 with weights pre-trained on ImageNet1K (version 1)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# For older PyTorch versions, you might see: model = resnet18(pretrained=True)
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Strategies for Transfer Learning</h3>
            
            <h4>1. Feature Extraction</h4>
            <p>
              <strong>Concept:</strong> Use the pre-trained model (specifically its convolutional base – all layers except the final classifier) as a fixed feature extractor. The idea is that the features learned on ImageNet are good enough.
            </p>
            <p><strong>Steps:</strong></p>
            <ol>
              <li>Load the pre-trained model.</li>
              <li><strong>Freeze the weights</strong> of all layers in the convolutional base. This means their weights won't be updated during training on your new dataset. You do this by setting <code>param.requires_grad = False</code> for each parameter in these layers.</li>
              <li>Replace the original final classification layer (which was trained for, say, 1000 ImageNet classes) with a new, randomly initialized classification layer suited for your task (e.g., 10 classes for CIFAR-10).</li>
              <li>Train <em>only</em> the parameters of this new classification layer on your dataset.</li>
            </ol>
            <p>
              <strong>When to Use:</strong> Good when your target dataset is small and/or very similar in nature to the dataset the model was originally trained on (e.g., ImageNet).
            </p>

            <h4>2. Fine-Tuning</h4>
            <p>
              <strong>Concept:</strong> Start with the pre-trained weights, but allow some or all of them to be adjusted (fine-tuned) for your new dataset.
            </p>
            <p><strong>Steps:</strong></p>
            <ol>
              <li>Load the pre-trained model.</li>
              <li>Replace the final classification layer with a new one for your task.</li>
              <li><strong>Unfreeze</strong> some or all of the layers in the pre-trained model (i.e., set <code>param.requires_grad = True</code>). You might choose to unfreeze only the later convolutional layers (which learn more task-specific features) or the entire network.</li>
              <li>Continue training the entire network (or the unfrozen parts) on your new dataset. It's crucial to use a <strong>very small learning rate</strong> for fine-tuning. This prevents the valuable pre-trained features from being distorted too quickly and drastically by large gradient updates from your (potentially small) new dataset.</li>
            </ol>
            <p>
              <strong>When to Use:</strong> Good when your target dataset is larger, or if it's somewhat different from the original training dataset, allowing the model to adapt its learned features more specifically.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Modifying the Classifier Head</h3>
            <p>
              Most pre-trained models from <code>torchvision.models</code> have a final fully connected layer (often named <code>fc</code> as in ResNet, or <code>classifier</code> as in VGG or AlexNet) that outputs scores for the original number of classes (e.g., 1000 for ImageNet). We need to replace this.
            </p>
            <pre><code class="language-python"> model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
 num_ftrs = model.fc.in_features # Get the number of input features to the original fc layer
 model.fc = nn.Linear(num_ftrs, NUM_CLASSES_OF_YOUR_DATASET) # Replace with a new fc layer
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Crucial: Input Image Transformations for Pre-trained Models</h3>
            <p>
              When using models pre-trained on ImageNet, you <strong>must</strong> preprocess your input images in the same way the ImageNet images were preprocessed during the original training.
            </p>
            <ol>
              <li><strong>Image Size:</strong> Most ImageNet models (like ResNet, VGG) expect input images of size <strong>224x224 pixels</strong>. CIFAR-10 images are 32x32. So, you'll need to add <code>transforms.Resize((224, 224))</code> to your transformation pipeline.</li>
              <li><strong>Normalization:</strong> You must normalize your images using the <strong>mean and standard deviation of the ImageNet dataset</strong>. These are standard values:
                <ul>
                  <li><code>mean = [0.485, 0.456, 0.406]</code> (for R, G, B channels)</li>
                  <li><code>std = [0.229, 0.224, 0.225]</code> (for R, G, B channels)</li>
                </ul>
                You'll use these in <code>transforms.Normalize(mean=mean, std=std)</code>.
              </li>
            </ol>
            <p>
              Data augmentation (like random flips, crops before resizing) can still be applied.
            </p>
          </div>

          <div class="lesson-section">
            <h3>Complete Code Example: Transfer Learning on CIFAR-10</h3>
            <p>
              Let's see the code for applying transfer learning (using ResNet18 as a feature extractor and then fine-tuning it) to CIFAR-10.
            </p>
            <pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18, ResNet18_Weights # Using specific weights enum
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time # To time epochs

# --- 1. Hyperparameters & Device Setup ---
# For Feature Extraction
FE_LEARNING_RATE = 0.001
FE_NUM_EPOCHS = 5 # Fewer epochs might be needed if only training the classifier

# For Fine-Tuning
FT_LEARNING_RATE = 0.0001 # Much smaller learning rate for fine-tuning
FT_NUM_EPOCHS = 10 # More epochs for fine-tuning the whole network

BATCH_SIZE = 64 # Can adjust based on GPU memory
MODEL_SAVE_PATH_FE = "cifar10_resnet18_ft_extractor.pth"
MODEL_SAVE_PATH_FT = "cifar10_resnet18_fine_tuned.pth"
CHECKPOINT_FILENAME_FE = "cifar10_resnet18_fe_checkpoint.pth"
CHECKPOINT_FILENAME_FT = "cifar10_resnet18_ft_checkpoint.pth"


# Determine device
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available. Number of GPUs: {num_gpus}")
    device = torch.device("cuda")
else:
    print("CUDA not available. Using CPU.")
    device = torch.device("cpu")
    num_gpus = 0

# --- 2. Data Loading & Transformation for CIFAR-10 (with ImageNet pre-training specs) ---
# ImageNet mean and std
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Transformations for pre-trained models
# Training: Resize, augment, ToTensor, Normalize
transform_train_pretrained = transforms.Compose([
    transforms.Resize(256), # Resize smaller edge to 256
    transforms.RandomResizedCrop(224), # Then take a random 224x224 crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Testing/Validation: Resize, CenterCrop, ToTensor, Normalize
transform_test_pretrained = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), # Take a center crop of 224x224
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

print("Loading CIFAR-10 dataset with ImageNet transformations...")
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train_pretrained)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 if num_gpus > 0 else 0)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test_pretrained)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 if num_gpus > 0 else 0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
NUM_CLASSES = len(classes)
print(f"CIFAR-10 dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# --- 3. Model Setup Function ---
def setup_model(num_classes, feature_extract_only=True):
    print(f"\nSetting up ResNet18 for {'Feature Extraction' if feature_extract_only else 'Fine-Tuning'}...")
    # Load pre-trained ResNet18 model
    model_tl = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Freeze parameters if doing feature extraction
    if feature_extract_only:
        print("Freezing parameters of convolutional base...")
        for param in model_tl.parameters():
            param.requires_grad = False # Freeze all layers initially

    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    num_ftrs = model_tl.fc.in_features # Get number of input features of the original fc layer
    model_tl.fc = nn.Linear(num_ftrs, num_classes) # Replace with new fc layer for NUM_CLASSES
    # For feature extraction, only model_tl.fc parameters will have requires_grad=True by default
    
    print(f"Replaced final layer. Output features: {num_classes}")
    return model_tl

# --- 4. Training and Evaluation Loop Function ---
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, 
                       checkpoint_filename, start_epoch=0, best_val_accuracy=0.0, model_type_name="Model"):
    
    print(f"\nStarting training for {model_type_name} from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images_val, labels_val in test_loader:
                images_val, labels_val = images_val.to(device), labels_val.to(device)
                outputs_val = model(images_val)
                loss_val = criterion(outputs_val, labels_val)
                running_val_loss += loss_val.item()
                _, predicted = torch.max(outputs_val.data, 1)
                val_total += labels_val.size(0)
                val_correct += (predicted == labels_val).sum().item()
        avg_val_loss = running_val_loss / len(test_loader)
        current_val_accuracy = 100 * val_correct / val_total
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {current_val_accuracy:.2f}% | Time: {epoch_duration:.2f}s")

        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            print(f"New best val acc: {best_val_accuracy:.2f}%. Saving checkpoint...")
            state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint = {
                'epoch': epoch + 1, 'model_state_dict': state_dict_to_save,
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_accuracy': best_val_accuracy
            }
            torch.save(checkpoint, checkpoint_filename)
        print("-" * 70)
    print(f"Finished training {model_type_name}. Best Val Acc: {best_val_accuracy:.2f}%")
    return model, best_val_accuracy


# --- Main Execution ---

# Part 1: Feature Extraction
# --------------------------
model_fe = setup_model(NUM_CLASSES, feature_extract_only=True)

if num_gpus > 1 and torch.cuda.is_available():
    model_fe = nn.DataParallel(model_fe)
model_fe.to(device)

# Observe that only parameters of the new classifier are being optimized
params_to_optimize_fe = []
print("\nParameters to optimize for Feature Extraction:")
for name, param in model_fe.named_parameters():
    if param.requires_grad:
        params_to_optimize_fe.append(param)
        print(f"\t{name}")
optimizer_fe = optim.Adam(params_to_optimize_fe, lr=FE_LEARNING_RATE)
criterion_fe = nn.CrossEntropyLoss()

model_fe, best_acc_fe = train_and_evaluate(model_fe, train_loader, test_loader, criterion_fe, optimizer_fe, 
                                           FE_NUM_EPOCHS, device, CHECKPOINT_FILENAME_FE, model_type_name="Feature Extractor")
torch.save(model_fe.module.state_dict() if isinstance(model_fe, nn.DataParallel) else model_fe.state_dict(), MODEL_SAVE_PATH_FE)
print(f"Feature extraction model saved to {MODEL_SAVE_PATH_FE}")


# Part 2: Fine-Tuning (Load the best feature extraction model and fine-tune it)
# -----------------------------------------------------------------------------
model_ft = setup_model(NUM_CLASSES, feature_extract_only=False) # Start with fresh ResNet for fine-tuning all layers

# Unfreeze all layers for fine-tuning (if they were frozen by setup_model's feature_extract_only=True path)
# The current setup_model with feature_extract_only=False already has all params with requires_grad=True
print("\nParameters to optimize for Fine-Tuning (all should be True):")
for name, param in model_ft.named_parameters():
     param.requires_grad = True # Ensure all are tunable
     # print(f"\t{name}: requires_grad={param.requires_grad}")


if num_gpus > 1 and torch.cuda.is_available():
    model_ft = nn.DataParallel(model_ft)
model_ft.to(device)


# Optimizer for fine-tuning: optimize all parameters with a smaller learning rate
optimizer_ft = optim.Adam(model_ft.parameters(), lr=FT_LEARNING_RATE)
criterion_ft = nn.CrossEntropyLoss()

model_ft, best_acc_ft = train_and_evaluate(model_ft, train_loader, test_loader, criterion_ft, optimizer_ft,
                                           FT_NUM_EPOCHS, device, CHECKPOINT_FILENAME_FT, model_type_name="Fine-Tuned Model")
torch.save(model_ft.module.state_dict() if isinstance(model_ft, nn.DataParallel) else model_ft.state_dict(), MODEL_SAVE_PATH_FT)
print(f"Fine-tuned model saved to {MODEL_SAVE_PATH_FT}")

print(f"\n--- Summary ---")
print(f"Best Validation Accuracy (Feature Extraction): {best_acc_fe:.2f}%")
print(f"Best Validation Accuracy (Fine-Tuning): {best_acc_ft:.2f}%")
</code></pre>
          </div>

          <div class="lesson-section">
            <h3>Try It Yourself</h3>
            <div class="challenge-box">
              <p>
                <strong>Task 5.2:</strong>
              </p>
              <ol>
                <li><strong>Study the Code:</strong>
                  <ul>
                    <li>Pay close attention to the <code>transform_train_pretrained</code> and <code>transform_test_pretrained</code>. Note the <code>Resize</code> to 224x224 and the use of <code>imagenet_mean</code> and <code>imagenet_std</code> for normalization. Why are these exact transformations critical?</li>
                    <li>Understand how <code>setup_model</code> function works, especially how <code>model_tl.fc</code> is replaced.</li>
                    <li>In the "Feature Extraction" part, notice how <code>param.requires_grad = False</code> is set for the pre-trained layers, and the optimizer is only given the parameters of the new <code>model_tl.fc</code> layer (or all layers that have <code>requires_grad=True</code>).</li>
                    <li>In the "Fine-Tuning" part, all parameters are made trainable (<code>param.requires_grad = True</code>), and the optimizer gets all <code>model_ft.parameters()</code>. Note the significantly smaller <code>FT_LEARNING_RATE</code>.</li>
                  </ul>
                </li>
                <li><strong>Run the Script:</strong> This will take longer to run than previous lessons, especially the fine-tuning part, as ResNet18 is a deeper model.
                  <ul>
                    <li>Execute the feature extraction part first. Observe its performance.</li>
                    <li>Then execute the fine-tuning part.</li>
                  </ul>
                </li>
                <li><strong>Observe and Analyze Performance:</strong>
                  <ul>
                    <li>How does the validation accuracy of the feature extraction model compare to the simple CNN you built from scratch in Lesson 5.1 for CIFAR-10? You should see a significant improvement.</li>
                    <li>How does the fine-tuned model's accuracy compare to the feature extraction model? Fine-tuning often yields further improvements if done carefully.</li>
                    <li>Notice the training times. Even though ResNet18 is larger, transfer learning can sometimes converge faster to a good solution.</li>
                  </ul>
                </li>
                <li><strong>(Optional Experimentation):</strong>
                  <ul>
                    <li>Try using a different pre-trained model, like <code>resnet34</code> (it will be slower to train but potentially more accurate). Remember to adjust the <code>weights</code> argument accordingly (e.g., <code>ResNet34_Weights.IMAGENET1K_V1</code>).</li>
                    <li>Experiment with the number of epochs for feature extraction vs. fine-tuning.</li>
                    <li>Instead of fine-tuning all layers, try to fine-tune only the last few convolutional blocks of ResNet along with the classifier. (This is more advanced, involving iterating through <code>model.named_parameters()</code> and selectively setting <code>requires_grad</code>).</li>
                  </ul>
                </li>
              </ol>
            </div>
          </div>
        `,
        completed: false,
      },
    ],
  },
];
