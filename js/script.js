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
      title: "Neural Networks",
      description:
        "Build and train basic neural networks using PyTorch's nn module.",
      lessonsCount: 6,
      progress: 0,
      difficulty: "Intermediate",
    },
    {
      id: 4,
      title: "Computer Vision",
      description:
        "Apply PyTorch to image classification, object detection, and other computer vision tasks.",
      lessonsCount: 7,
      progress: 0,
      difficulty: "Intermediate",
    },
    {
      id: 5,
      title: "Natural Language Processing",
      description:
        "Process and analyze text data using PyTorch for tasks like sentiment analysis.",
      lessonsCount: 6,
      progress: 0,
      difficulty: "Advanced",
    },
    {
      id: 6,
      title: "Deployment & Optimization",
      description:
        "Learn how to deploy PyTorch models to production and optimize their performance.",
      lessonsCount: 5,
      progress: 0,
      difficulty: "Advanced",
    },
  ];

  // Update page titles and metadata
  document.title = document.title.replace("PyTorch Learning Platform", "Learn PyTorch, fast.");
  
  // Replace platform name in footers
  document.querySelectorAll('.footer-logo p').forEach(el => {
    el.textContent = "PyTorch Fundamentals";
  });
  
  document.querySelectorAll('.footer-bottom p').forEach(el => {
    el.innerHTML = '&copy; 2023 PyTorch Fundamentals by <a href="https://sohambuilds.github.io" target="_blank">Soham Roy</a>';
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
