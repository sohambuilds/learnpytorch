document.addEventListener("DOMContentLoaded", () => {
  // Theme switcher
  const toggleSwitch = document.querySelector(
    '.theme-switch input[type="checkbox"]'
  );

  function switchTheme(e) {
    if (e.target.checked) {
      document.documentElement.setAttribute("data-theme", "dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.setAttribute("data-theme", "light");
      localStorage.setItem("theme", "light");
    }
  }

  toggleSwitch.addEventListener("change", switchTheme, false);

  // Make sure dark mode is set as default unless explicitly changed by user
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme === "light") {
    document.documentElement.setAttribute("data-theme", "light");
    toggleSwitch.checked = false;
  } else {
    // Either dark mode is saved or no preference is stored (default to dark)
    document.documentElement.setAttribute("data-theme", "dark");
    toggleSwitch.checked = true;
  }

  // Mobile menu toggle
  const menuBtn = document.getElementById("mobile-menu-btn");
  const mainNav = document.getElementById("main-nav");

  if (menuBtn) {
    menuBtn.addEventListener("click", () => {
      mainNav.classList.toggle("active");
      menuBtn.classList.toggle("open");
    });
  }

  // Sample module data - this would typically come from a JSON file or API
  const modules = [
    {
      id: 1,
      title: "Introduction to PyTorch",
      description:
        "Learn the basics of PyTorch, its architecture, and how it compares to other frameworks.",
      lessonsCount: 4,
      progress: 0,
      difficulty: "Beginner",
    },
    {
      id: 2,
      title: "Building Neural Networks",
      description:
        "This module transitions from basic tensor operations to constructing neural networks. Learn how to build, structure, and understand neural networks using PyTorch's nn module, including layers, activations, and model containers.",
      lessonsCount: 5,
      progress: 0,
      difficulty: "Intermediate",
    },
    {
      id: 3,
      title: "Training",
      description: `Learn how to efficiently load data with Datasets and DataLoaders, 
                    implement a complete training loop, and properly evaluate model 
                    performance on unseen data. You'll master data batching, training 
                    loop components (forward pass, loss calculation, backpropagation), 
                    and validation techniques.`,
      lessonsCount: 3,
      progress: 0,
      difficulty: "Intermediate",
    },
    {
      id: 4,
      title: "Practical Considerations",
      description:
        "Learn key practical skills for PyTorch development, including saving models and GPU acceleration.",
      lessonsCount: 3,
      progress: 0,
      difficulty: "Intermediate",
    },
    {
      id: 5,
      title: "Computer Vision with PyTorch",
      description:
        "Build a CNN in PyTorch for CIFAR-10 dataset. Learn how to use transfer learning to build a more complex model.",
      lessonsCount: 3,
      progress: 0,
      difficulty: "Intermediate",
    },
    {
      id: 6,
      title: "NLP with PyTorch",
      description:
        "Leaarn the basics of Natural Language Processing. Build an RNN and LSTM for IMDB dataset and compare the performance.  ",
      lessonsCount: 5,
      progress: 0,
      difficulty: "Advanced",
    },
  ];

  // Update page titles and metadata
  document.title = document.title.replace(
    "PyTorch Learning Platform",
    "Learn PyTorch, fast."
  );

  // Replace platform name in footers
  document.querySelectorAll(".footer-logo p").forEach((el) => {
    el.textContent = "PyTorch Fundamentals";
  });

  // Replace copyright year in footer
  document.querySelectorAll(".footer-bottom p").forEach((el) => {
    el.innerHTML =
      '&copy; 2025 PyTorch Fundamentals by <a href="https://sohambuilds.github.io" target="_blank" style="color:#ff5722;">Soham Roy</a>';
  });

  // Load progress from localStorage
  modules.forEach((module) => {
    const savedProgress = localStorage.getItem(`module-${module.id}-progress`);
    if (savedProgress) {
      module.progress = parseInt(savedProgress);
    }
  });

  // Render module cards on the homepage
  const modulePreview = document.querySelector(".module-preview .module-cards");
  if (modulePreview) {
    // Add fade-in animation to module cards
    let delay = 0;
    // Show only first 3 modules on homepage
    modules.slice(0, 3).forEach((module) => {
      const card = createModuleCard(module);
      card.style.animation = `fadeIn 0.5s ease-out ${delay}s forwards`;
      card.style.opacity = "0";
      modulePreview.appendChild(card);
      delay += 0.2;
    });
  }

  // Render all module cards on the modules page
  const modulesList = document.getElementById("modules-list");
  if (modulesList) {
    let delay = 0;
    modules.forEach((module) => {
      const card = createModuleCard(module);
      card.style.animation = `fadeIn 0.5s ease-out ${delay}s forwards`;
      card.style.opacity = "0";
      modulesList.appendChild(card);
      delay += 0.1;
    });
  }

  // Create a module card element
  function createModuleCard(module) {
    const card = document.createElement("div");
    card.className = "module-card";

    // Add difficulty badge
    const difficultyClass = module.difficulty.toLowerCase();

    card.innerHTML = `
            <span class="badge ${difficultyClass}">${module.difficulty}</span>
            <h3>${module.title}</h3>
            <p class="lessons-count">${module.lessonsCount} Lessons</p>
            <p class="description">${module.description}</p>
            <div class="progress-container">
                <div class="progress-bar" style="width: ${module.progress}%"></div>
            </div>
            <p class="progress-text">${module.progress}% Complete</p>
        `;

    // Add click event to navigate to module page
    card.addEventListener("click", () => {
      window.location.href = `module.html?id=${module.id}`;
    });

    return card;
  }
});
