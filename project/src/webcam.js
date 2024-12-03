export class WebcamManager {
  constructor() {
    this.video = document.getElementById('webcam');
    this.stream = null;
  }

  async setup(deviceId = null) {
    try {
      // Stop any existing stream
      this.stop();

      const constraints = {
        video: deviceId ? 
          {
            deviceId: { exact: deviceId },
            width: { ideal: 640 },
            height: { ideal: 480 }
          } : 
          {
            width: { ideal: 640 },
            height: { ideal: 480 }
          },
        audio: false
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;
      
      // Wait for video metadata to be loaded before setting canvas dimensions
      await new Promise((resolve) => {
        this.video.onloadedmetadata = () => {
          const canvas = document.getElementById('overlay');
          canvas.width = this.video.videoWidth;
          canvas.height = this.video.videoHeight;
          resolve();
        };
      });
      
      return true;
    } catch (error) {
      console.error('Error accessing webcam:', error);
      return false;
    }
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.video.srcObject = null;
      this.stream = null;
    }
  }

  isActive() {
    return !!this.stream;
  }
}