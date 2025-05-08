// This file now only aggregates all modules from separate files.

window.modulesData = [
  ...(window.module1Data || []),
  ...(window.module2Data || []),
  ...(window.module3Data || []),
  ...(window.module4Data || []),
  ...(window.module5Data || []),
  ...(window.module6Data || []),
  // ...add more as you create more modules
];
