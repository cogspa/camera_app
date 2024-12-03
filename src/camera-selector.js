export class CameraSelector {
  constructor() {
    this.devices = [];
  }

  async getAvailableCameras() {
    try {
      // Check if mediaDevices is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        throw new Error('Media devices API not supported');
      }

      // Request permission with explicit video constraints
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });

      // Stop the temporary stream immediately
      stream.getTracks().forEach(track => track.stop());
      
      // Now that we have permission, enumerate devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      this.devices = devices.filter(device => device.kind === 'videoinput');
      
      return this.devices;
    } catch (error) {
      console.error('Error getting camera devices:', error);
      return [];
    }
  }

  createCameraSelect(container) {
    // Clear existing content
    container.innerHTML = '';
    
    const select = document.createElement('select');
    select.className = 'mt-2 px-4 py-2 bg-gray-800 rounded text-white';
    select.id = 'camera-select';
    
    if (this.devices.length === 0) {
      const option = document.createElement('option');
      option.value = '';
      option.text = 'No cameras found';
      select.appendChild(option);
      select.disabled = true;
    } else {
      this.devices.forEach((device, index) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text = device.label || `Camera ${index + 1}`;
        select.appendChild(option);
      });
    }

    container.appendChild(select);
    return select;
  }
}