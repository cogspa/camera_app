import { WebcamManager } from './webcam';
import { FaceTracker } from './face-tracker';
import { CameraSelector } from './camera-selector';
import { DiagnosticPanel } from './components/diagnostic-panel';

class App {
  constructor() {
    this.webcamManager = new WebcamManager();
    this.faceTracker = new FaceTracker(
      document.getElementById('webcam'),
      document.getElementById('overlay')
    );
    this.cameraSelector = new CameraSelector();
    
    this.startBtn = document.getElementById('startBtn');
    this.stopBtn = document.getElementById('stopBtn');
    this.cameraSelectContainer = document.getElementById('camera-select-container');
    
    // Create diagnostic panel
    const mainContainer = document.querySelector('.max-w-3xl');
    this.diagnosticPanel = new DiagnosticPanel(mainContainer);
    
    // Wait for DOM to be fully loaded
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.init());
    } else {
      this.init();
    }
  }

  async init() {
    // Run diagnostics first
    await this.diagnosticPanel.runDiagnostics();
    await this.initializeCameras();
    this.setupEventListeners();
  }

  async initializeCameras() {
    try {
      const cameras = await this.cameraSelector.getAvailableCameras();
      if (cameras.length > 0) {
        const select = this.cameraSelector.createCameraSelect(this.cameraSelectContainer);
        select.addEventListener('change', () => {
          if (this.webcamManager.isActive()) {
            this.stop();
            this.start();
          }
        });
      } else {
        this.startBtn.disabled = true;
        this.cameraSelectContainer.innerHTML = '<p class="text-red-500">No cameras detected. Please connect a camera and refresh the page.</p>';
      }
    } catch (error) {
      console.error('Failed to initialize cameras:', error);
      this.startBtn.disabled = true;
      this.cameraSelectContainer.innerHTML = '<p class="text-red-500">Failed to access camera. Please ensure you have granted camera permissions.</p>';
    }
  }

  setupEventListeners() {
    this.startBtn.addEventListener('click', () => this.start());
    this.stopBtn.addEventListener('click', () => this.stop());
  }

  async start() {
    this.startBtn.disabled = true;
    
    const selectedCamera = document.getElementById('camera-select')?.value;
    const webcamStarted = await this.webcamManager.setup(selectedCamera);
    
    if (!webcamStarted) {
      alert('Failed to access webcam. Please ensure you have granted camera permissions.');
      this.startBtn.disabled = false;
      return;
    }

    const modelLoaded = await this.faceTracker.initialize();
    if (!modelLoaded) {
      alert('Failed to load face tracking model. Please try again.');
      this.webcamManager.stop();
      this.startBtn.disabled = false;
      return;
    }

    this.faceTracker.startTracking();
    this.stopBtn.disabled = false;
  }

  stop() {
    this.faceTracker.stopTracking();
    this.webcamManager.stop();
    this.startBtn.disabled = false;
    this.stopBtn.disabled = true;
  }
}

// Initialize the application
new App();