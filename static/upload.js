const form = document.getElementById('uploadForm');
const imageFile = document.getElementById('imageFile');
const imagePreview = document.getElementById('imagePreview');  // Add an element for the image preview

form.addEventListener('submit', function(event) {
  event.preventDefault(); // Prevent default form submission

  // Check if a file is selected
  if (!imageFile.files.length) {
    alert('Please select an image to upload.');
    return;
  }

  const formData = new FormData();
  formData.append('image', imageFile.files[0]);

  fetch('/upload_image', {  // Replace with your Flask endpoint URL
    method: 'POST',
    body: formData
  })
});
