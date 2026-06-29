document.addEventListener("DOMContentLoaded", () => {
  const setLoading = (form, label) => {
    form.addEventListener("submit", () => {
      const btn = form.querySelector('button[type="submit"]');
      if (btn) {
        btn.disabled = true;
        btn.textContent = label;
      }
    });
  };

  document.querySelectorAll("form.prediction-form").forEach((form) => {
    setLoading(form, "Predicting...");
  });

  document.querySelectorAll("form.dataset-manual-form").forEach((form) => {
    setLoading(form, "Adding...");
  });

  document.querySelectorAll("form.dataset-upload-form").forEach((form) => {
    setLoading(form, "Uploading...");
  });

  const fileInput = document.querySelector('input[name="dataset_file"]');
  const uploadZone = document.querySelector(".upload-zone");
  if (fileInput && uploadZone) {
    uploadZone.addEventListener("dragover", (event) => {
      event.preventDefault();
      uploadZone.style.borderColor = "rgba(62, 198, 224, 0.7)";
    });

    uploadZone.addEventListener("dragleave", () => {
      uploadZone.style.borderColor = "";
    });

    uploadZone.addEventListener("drop", (event) => {
      event.preventDefault();
      uploadZone.style.borderColor = "";
      if (event.dataTransfer.files.length) {
        fileInput.files = event.dataTransfer.files;
      }
    });
  }
});
