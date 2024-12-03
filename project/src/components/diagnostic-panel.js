import { CameraDiagnostics } from '../utils/diagnostics';

export class DiagnosticPanel {
  constructor(container) {
    this.container = container;
    this.panel = document.createElement('div');
    this.panel.className = 'bg-gray-800 p-4 rounded-lg mb-4';
    this.container.insertBefore(this.panel, this.container.firstChild);
  }

  async runDiagnostics() {
    this.panel.innerHTML = '<p class="text-yellow-400">Running diagnostics...</p>';

    // Check browser support
    const browserSupport = await CameraDiagnostics.checkBrowserSupport();
    const permissionStatus = await CameraDiagnostics.checkCameraPermission();
    const cameraStatus = await CameraDiagnostics.checkConnectedCameras();

    let html = '<div class="space-y-2">';
    
    // Browser Support Section
    html += '<div class="mb-3">';
    html += '<h3 class="font-bold mb-2">Browser Compatibility:</h3>';
    if (browserSupport.errors.length === 0) {
      html += '<p class="text-green-400">✓ Your browser fully supports all required features</p>';
    } else {
      html += '<div class="text-red-400">';
      browserSupport.errors.forEach(error => {
        html += `<p>✗ ${error}</p>`;
      });
      html += '</div>';
    }
    html += '</div>';

    // Camera Permission Section
    html += '<div class="mb-3">';
    html += '<h3 class="font-bold mb-2">Camera Permission:</h3>';
    if (permissionStatus.granted) {
      html += '<p class="text-green-400">✓ Camera permission granted</p>';
    } else {
      html += `<p class="text-red-400">✗ ${permissionStatus.error}</p>`;
      if (permissionStatus.error === 'Camera permission denied') {
        html += `<p class="text-sm mt-1">To fix: Click the camera icon in your browser's address bar and allow access</p>`;
      }
    }
    html += '</div>';

    // Connected Cameras Section
    html += '<div class="mb-3">';
    html += '<h3 class="font-bold mb-2">Connected Cameras:</h3>';
    if (cameraStatus.found) {
      html += `<p class="text-green-400">✓ Found ${cameraStatus.count} camera${cameraStatus.count > 1 ? 's' : ''}</p>`;
    } else {
      html += `<p class="text-red-400">✗ ${cameraStatus.error || 'No cameras detected'}</p>`;
      html += '<p class="text-sm mt-1">Make sure a camera is properly connected to your computer</p>';
    }
    html += '</div>';

    this.panel.innerHTML = html;
  }
}