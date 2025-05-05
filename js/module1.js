window.module1Data = [
  {
    id: 1,
    title: "Introduction to PyTorch",
    description:
      "Learn the basics of PyTorch, its architecture, and how it compares to other frameworks.",
    lessons: [
      {
        id: 1,
        title: "What is PyTorch?",
        content: `
              <h2>What is PyTorch?</h2>
              <p>Hey there! Let's get straight to the point: PyTorch is one of the most popular deep learning frameworks out there, but what exactly makes it so special?</p>

              <div class="lesson-section">
                <h3>PyTorch in a Nutshell</h3>
                <p>At its core, PyTorch is two things:</p>
                <ol>
                  <li>A powerful library for working with tensors (think of them as souped-up arrays)</li>
                  <li>A platform for building and training neural networks</li>
                </ol>
                
                <p>If you're coming from a data science background, you might be thinking, "Isn't this just NumPy with a fancy name?" Great question! While there are similarities, PyTorch offers something crucial that NumPy doesn't: <strong>GPU acceleration</strong> and <strong>automatic differentiation</strong>.</p>
              </div>

              <div class="lesson-section">
                <h3>The Tensor Library: Speed When You Need It</h3>
                <p>Tensors in PyTorch are similar to NumPy arrays, but with a major advantage: they can run on your GPU. Why does this matter? Because GPUs can perform certain calculations (especially the matrix math that deep learning loves) dramatically faster than CPUs.</p>
                
                <pre><code class="language-python">import torch

# Create a tensor on CPU
cpu_tensor = torch.tensor([1, 2, 3])
print(f"Tensor on CPU: {cpu_tensor}")

# Move it to GPU (if available)
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.cuda()
    print(f"Same tensor on GPU: {gpu_tensor}")
    print(f"Is it still the same data? {torch.equal(cpu_tensor.cpu(), gpu_tensor.cpu())}")
else:
    print("No GPU available - but your code would be much faster if there was!")</code></pre>
              </div>

              <div class="lesson-section">
                <h3>Autograd: Calculus Without the Headaches</h3>
                <p>Here's where PyTorch really shines. Deep learning requires calculating gradients to update model parameters, which traditionally meant doing a lot of calculus by hand or deriving complex equations.</p>
                
                <p>PyTorch's Autograd system tracks the operations you perform and automatically calculates these gradients for you. It's like having a math genius constantly looking over your shoulder, figuring out all the derivatives you need.</p>
                
                <p>I'll show you Autograd in action in the next lesson, but trust me, it makes your life infinitely easier.</p>
              </div>

              <div class="lesson-section">
                <h3>Why Use a Framework Like PyTorch?</h3>
                <p>Imagine building a house. You could theoretically make your own bricks, forge your own nails, and cut your own lumber... or you could just use pre-made materials and focus on designing the house itself.</p>
                
                <p>PyTorch gives you the building blocks for deep learning so you can focus on the creative and challenging parts - designing models and solving problems - rather than reinventing the wheel with implementation details.</p>
                
                <p>Plus, PyTorch is:</p>
                <ul>
                  <li><strong>Pythonic</strong> - it feels natural if you know Python</li>
                  <li><strong>Dynamic</strong> - you can change your network on the fly</li>
                  <li><strong>Debuggable</strong> - standard Python debugging tools work</li>
                </ul>
              </div>

              <div class="lesson-section">
                <h3>Think About It</h3>
                <p>Imagine you're trying to optimize a function with thousands or millions of parameters. For each parameter, you need to calculate how a small change affects the output.</p>
                
                <p>Without automatic differentiation, you'd need to:</p>
                <ol>
                  <li>Derive each partial derivative by hand</li>
                  <li>Code each one individually</li>
                  <li>Test and debug each implementation</li>
                  <li>Hope you don't make any calculus mistakes</li>
                </ol>
                
                <p>How many hours would that take? How many opportunities for errors? This is why Autograd is such a game-changer for deep learning.</p>
                
                <div class="challenge-box">
                  <p><strong>Task:</strong> Think about a simple function like y = w×x + b. In machine learning, we adjust w and b to make y match a target value. This requires calculating derivatives of an error like (y−target)² with respect to w and b.</p>
                  <p>Now imagine a network with millions of parameters. Why would manually calculating all these derivatives be nearly impossible and highly error-prone?</p>
                </div>
              </div>
            `,
        completed: false,
      },
      {
        id: 2,
        title: "Tensors - The Building Blocks",
        content: `
              <h2>Tensors - The Building Blocks</h2>
              <p>
                Tensors are the heart of PyTorch. If you get comfortable with them, everything else gets easier. Let's break down what they are and how to use them in practice.
              </p>

              <div class="lesson-section">
                <h3>What is a Tensor?</h3>
                <ul>
                  <li><strong>0D:</strong> A single number (scalar)</li>
                  <li><strong>1D:</strong> A list of numbers (vector)</li>
                  <li><strong>2D:</strong> A table of numbers (matrix)</li>
                  <li><strong>3D+:</strong> Think images, batches, or even crazier stuff</li>
                </ul>
                <p>
                  If you've used NumPy arrays, tensors will feel familiar. The big difference? <strong>PyTorch tensors can use your GPU for speed.</strong>
                </p>
              </div>

              <div class="lesson-section">
                <h3>Creating Tensors</h3>
                <p>Here are some common ways to make tensors:</p>
                <pre><code class="language-python">import torch
import numpy as np

# From a Python list
t1 = torch.tensor([[1, 2], [3, 4]])
print(t1)
# tensor([[1, 2],
#         [3, 4]])

# From a NumPy array (shares memory if on CPU)
arr = np.array([[5., 6.], [7., 8.]])
t2 = torch.from_numpy(arr)
print(t2)
# tensor([[5., 6.],
#         [7., 8.]])

# With built-in functions
t_rand = torch.randn(2, 3)  # random normal
t_zeros = torch.zeros(3, 2)
t_ones = torch.ones(4)
print(t_rand)
print(t_zeros)
print(t_ones)
</code></pre>
                <p><strong>Try it yourself:</strong> Make a tensor of shape (2, 5) filled with ones.</p>
              </div>

              <div class="lesson-section">
                <h3>Tensor Properties</h3>
                <p>Every tensor has a few key properties:</p>
                <ul>
                  <li><code>.shape</code> — the size in each dimension</li>
                  <li><code>.dtype</code> — data type (float, int, etc.)</li>
                  <li><code>.device</code> — CPU or GPU</li>
                </ul>
                <pre><code class="language-python">t = torch.randn(3, 4)
print(t.shape)   # torch.Size([3, 4])
print(t.dtype)   # torch.float32 (usually)
print(t.device)  # cpu or cuda:0
</code></pre>
              </div>

              <div class="lesson-section">
                <h3>Indexing, Slicing, and Reshaping</h3>
                <p>Just like NumPy, you can grab parts of tensors or change their shape:</p>
                <pre><code class="language-python">t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t[0])      # First row: tensor([1, 2, 3])
print(t[:, 1])   # Second column: tensor([2, 5])

# Reshape (must have same number of elements)
flat = torch.arange(1, 7)
print(flat.reshape(2, 3))
# tensor([[1, 2, 3],
#         [4, 5, 6]])
</code></pre>
                <p><strong>Try it yourself:</strong> Slice out the last column from a 3x3 tensor.</p>
              </div>

              <div class="lesson-section">
                <h3>Operations</h3>
                <p>Most math works as you'd expect:</p>
                <pre><code class="language-python">a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

print(a + b)      # Element-wise sum
print(a * b)      # Element-wise product
print(a @ b)      # Matrix multiplication
</code></pre>
                <p>
                  <strong>Note:</strong> <code>@</code> is matrix multiply, not element-wise!
                </p>
              </div>

              <div class="lesson-section">
                <h3>PyTorch & NumPy: Best Friends</h3>
                <p>
                  You can convert between PyTorch tensors and NumPy arrays easily (as long as the tensor is on the CPU). They often share memory, so changing one can change the other!
                </p>
                <pre><code class="language-python">import numpy as np
t = torch.ones(3)
arr = t.numpy()
arr[0] = 99
print(t)  # tensor([99.,  1.,  1.])
</code></pre>
              </div>

              <div class="lesson-section">
                <h3>Recap & Challenge</h3>
                <ul>
                  <li>Tensors are the core data structure in PyTorch</li>
                  <li>They're like NumPy arrays, but with GPU support and more</li>
                  <li>Get comfortable with creation, indexing, reshaping, and math</li>
                </ul>
                <div class="challenge-box">
                  <p><strong>Challenge:</strong> Create a 2x3 tensor of random numbers and print its mean. Then, convert it to a NumPy array and add 1 to every element. What happens to the tensor?</p>
                </div>
              </div>
            `,
        completed: false,
      },
      {
        id: 3,
        title: "Automatic Differentiation (Autograd)",
        content: `
              <h2>Autograd - Automatic Differentiation</h2>
              <p>
                <strong>Autograd</strong> is PyTorch's secret sauce for deep learning. It automatically keeps track of all the operations you do on tensors and builds a "computational graph" behind the scenes. This graph lets PyTorch figure out how to compute gradients for you, so you never have to do calculus by hand.
              </p>
              <div class="lesson-section">
                <h3>How Does Autograd Actually Work?</h3>
                <ul>
                  <li>
                    <strong>Computational Graph:</strong> Imagine every tensor as a node in a graph. Every operation you do (like <code>+</code>, <code>*</code>, <code>**</code>, etc.) adds edges and new nodes. This graph records how each value was computed from others.
                  </li>
                  <li>
                    <strong>Why a Graph?</strong> When you want to know how a final value (like your loss) changes with respect to your parameters, PyTorch can "walk backward" through this graph using the chain rule from calculus. This is called <strong>backpropagation</strong>.
                  </li>
                  <li>
                    <strong>requires_grad=True:</strong> If you want PyTorch to track gradients for a tensor (like weights or inputs you want to optimize), set <code>requires_grad=True</code> when you create it. Any operation involving such a tensor will also be tracked.
                  </li>
                  <li>
                    <strong>Leaf nodes and intermediate nodes:</strong> Tensors you create directly (like model parameters) are "leaf nodes." Tensors created by operations (like <code>y = w * x + b</code>) are intermediate nodes. Gradients are only stored for leaf nodes with <code>requires_grad=True</code>.
                  </li>
                  <li>
                    <strong>.backward():</strong> When you have a scalar output (like a loss), call <code>.backward()</code> on it. PyTorch will compute all the gradients for you, filling the <code>.grad</code> attribute of each leaf tensor.
                  </li>
                  <li>
                    <strong>Why only scalars?</strong> PyTorch needs a single number to start the chain rule. If you have a vector or matrix, you must specify which direction you want the gradient in (rare for most deep learning).
                  </li>
                  <li>
                    <strong>Accumulation:</strong> Gradients accumulate by default. Always zero them out before the next backward pass, or you'll get weird results!
                  </li>
                  <li>
                    <strong>Turning off autograd:</strong> For inference or when you don't need gradients, use <code>with torch.no_grad():</code> or <code>.detach()</code> to save memory and speed things up.
                  </li>
                </ul>
                <p>
                  <em>Bottom line:</em> You focus on building your model and loss function. PyTorch handles the math for updating your parameters.
                </p>
              </div>

              <div class="lesson-section">
                <h3>Basic Autograd Example</h3>
                <pre><code class="language-python">import torch

# Create tensors that require gradients
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Simple calculation (our "model")
y = w * x + b  # y = 3 * 2 + 1 = 7

# Compute gradients
y.backward()

# Gradients are now stored in .grad
print(f"dy/dx: {x.grad}")  # 3
print(f"dy/dw: {w.grad}")  # 2
print(f"dy/db: {b.grad}")  # 1
</code></pre>
                <p>
                  <strong>Notice:</strong> The gradients match the partial derivatives you'd expect from calculus!
                </p>
              </div>

              <div class="lesson-section">
                <h3>Gradients Accumulate</h3>
                <p>
                  If you call <code>.backward()</code> again without zeroing gradients, PyTorch adds the new gradients to the old ones.
                </p>
                <pre><code class="language-python"># Another output using the same parameters
z = w * x**2  # z = 3 * 2^2 = 12
z.backward()

print("--- After second backward() call ---")
print(f"x.grad: {x.grad}")  # 3 (from y) + 12 (from z) = 15
print(f"w.grad: {w.grad}")  # 2 (from y) + 4 (from z) = 6
print(f"b.grad: {b.grad}")  # 1 (from y) + 0 (from z) = 1
</code></pre>
                <p>
                  <strong>Tip:</strong> Always zero gradients before the next backward pass!
                </p>
                <pre><code class="language-python">x.grad.zero_()
w.grad.zero_()
b.grad.zero_()
print("--- After zeroing gradients ---")
print(f"x.grad: {x.grad}")
print(f"w.grad: {w.grad}")
print(f"b.grad: {b.grad}")
</code></pre>
              </div>

              <div class="lesson-section">
                <h3>Disabling Gradient Tracking</h3>
                <p>
                  During inference or when you don't need gradients, turn them off for efficiency.
                </p>
                <pre><code class="language-python">with torch.no_grad():
    k = w * x + b
    print(f"k = {k}")
    print(f"Does k require grad? {k.requires_grad}")  # False

# Or use .detach() to get a tensor with no grad history
y = w * x + b
y_detached = y.detach()
print(f"y_detached requires grad: {y_detached.requires_grad}")  # False
</code></pre>
              </div>

              <div class="lesson-section">
                <h3>Try It Yourself</h3>
                <div class="challenge-box">
                  <p>
                    <strong>Task:</strong> Create three tensors <code>a</code>, <code>b</code>, and <code>c</code> with scalar values (e.g., a=2.0, b=4.0, c=3.0), all requiring gradients.<br>
                    Define <code>d = a * b + c ** 2</code>.<br>
                    Before calling <code>.backward()</code>, predict what the gradients <code>∂d/∂a</code>, <code>∂d/∂b</code>, and <code>∂d/∂c</code> should be.<br>
                    Call <code>.backward()</code> on <code>d</code>.<br>
                    Print the <code>.grad</code> attributes of <code>a</code>, <code>b</code>, and <code>c</code>. Do they match your prediction?
                  </p>
                  <pre><code class="language-python">import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
c = torch.tensor(3.0, requires_grad=True)

d = a * b + c ** 2
d.backward()

print(f"∂d/∂a: {a.grad}")  # Should be b = 4
print(f"∂d/∂b: {b.grad}")  # Should be a = 2
print(f"∂d/∂c: {c.grad}")  # Should be 2 * c = 6
</code></pre>
                </div>
              </div>
            `,
        completed: false,
      },
      {
        id: 4,
        title: "Your First PyTorch Program",
        content: `
              <h2>Your First PyTorch Program: Linear Regression from Scratch</h2>
              <p>
                Let's put together everything you've learned so far! We'll build and train a simple linear regression model using PyTorch tensors and autograd—no fancy libraries, just the basics.
              </p>
              <div class="lesson-section">
                <h3>What Are We Doing?</h3>
                <ul>
                  <li>
                    <strong>Linear regression</strong> is the simplest way to predict a number from another number. Imagine you want to predict someone's weight from their height, or the price of a house from its size. The relationship is a straight line: <code>y = wx + b</code>.
                  </li>
                  <li>
                    We'll create some fake data that follows a line: <code>y = 2x + 1</code> (with a little noise, to make it realistic).
                  </li>
                  <li>
                    We'll start with random guesses for <code>w</code> and <code>b</code> (the slope and intercept).
                  </li>
                  <li>
                    We'll use PyTorch to predict <code>y</code> for each <code>x</code>, measure how "off" our guess is (the <strong>loss</strong>), and then use <strong>autograd</strong> to figure out how to adjust <code>w</code> and <code>b</code> to make the predictions better.
                  </li>
                  <li>
                    We'll repeat this process (called <strong>training</strong>) and watch our model get better at fitting the data.
                  </li>
                </ul>
                <p>
                  <em>Don't worry if this sounds new—just follow along and you'll see how it works!</em>
                </p>
              </div>
              <div class="lesson-section">
                <h3>Step-by-Step Example</h3>
                <pre><code class="language-python">import torch

# 1. Create some fake data: y = 2x + 1 + noise
torch.manual_seed(42)  # For reproducibility
X = torch.linspace(0, 10, 20).unsqueeze(1)  # 20 points between 0 and 10, shape (20, 1)
y = 2 * X + 1 + torch.randn(X.size()) * 2   # Add some random noise

# 2. Initialize weights and bias (randomly)
w = torch.randn(1, requires_grad=True)  # Our guess for the slope
b = torch.randn(1, requires_grad=True)  # Our guess for the intercept

learning_rate = 0.03

for step in range(50):
    # 3. Forward pass: predict y using our current w and b
    y_pred = w * X + b

    # 4. Compute mean squared error loss (how far off are we?)
    loss = ((y_pred - y) ** 2).mean()

    # 5. Backward pass: compute gradients of loss w.r.t. w and b
    loss.backward()

    # 6. Update weights and bias using the gradients (gradient descent)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 7. Zero gradients for next step (important!)
    w.grad.zero_()
    b.grad.zero_()

    if step % 10 == 0:
        print(f"Step {step}: loss={loss.item():.4f}, w={w.item():.2f}, b={b.item():.2f}")

print(f"Final learned parameters: w={w.item():.2f}, b={b.item():.2f}")
</code></pre>
                <p>
                  <strong>What to notice:</strong>
                  <ul>
                    <li>We use <code>requires_grad=True</code> for parameters we want to learn.</li>
                    <li>We call <code>loss.backward()</code> to compute gradients automatically.</li>
                    <li>We update parameters inside <code>torch.no_grad()</code> so PyTorch doesn't track those ops.</li>
                    <li>We zero gradients after each update (otherwise they accumulate).</li>
                    <li>After a few steps, <code>w</code> and <code>b</code> should get close to 2 and 1—the true values!</li>
                  </ul>
                </p>
              </div>
              <div class="lesson-section">
                <h3>What's Happening Under the Hood?</h3>
                <ul>
                  <li>
                    <strong>Forward pass:</strong> We use our current guess for <code>w</code> and <code>b</code> to make predictions.
                  </li>
                  <li>
                    <strong>Loss:</strong> We measure how far off our predictions are from the real data. (Mean squared error is just the average of the squared differences.)
                  </li>
                  <li>
                    <strong>Backward pass:</strong> PyTorch uses autograd to figure out how to change <code>w</code> and <code>b</code> to make the loss smaller.
                  </li>
                  <li>
                    <strong>Update:</strong> We nudge <code>w</code> and <code>b</code> in the direction that reduces the loss.
                  </li>
                  <li>
                    <strong>Repeat:</strong> Each time, our model gets a little better at fitting the data.
                  </li>
                </ul>
                <p>
                  This is the basic idea behind all deep learning: make a guess, measure how wrong you are, use gradients to improve, and repeat!
                </p>
              </div>
              <div class="lesson-section">
                <h3>Try It Yourself</h3>
                <div class="challenge-box">
                  <p>
                    <strong>Challenge:</strong> Change the true relationship to <code>y = -3x + 5</code> and see if your model learns the new weights and bias. Try changing the learning rate or number of steps—what happens? (Tip: If the loss doesn't go down, try a smaller learning rate!)
                  </p>
                </div>
              </div>
              <div class="lesson-section">
                <h3>Why This Matters</h3>
                <p>
                  This simple example is the foundation of all machine learning and deep learning. If you understand this loop—predict, measure, compute gradients, update—you can understand how neural networks learn!
                </p>
                <p>
                  In the next modules, you'll see how PyTorch makes it even easier to build more complex models.
                </p>
              </div>
            `,
        completed: false,
      },
    ],
  },
];
