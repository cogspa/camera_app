export class CameraDiagnostics {
  static async checkBrowserSupport() {
    const results = {
      mediaDevicesSupported: false,
      getUserMediaSupported: false,
      enumerateDevicesSupported: false,
      errors: []
    };

    // Check MediaDevices API support
    if (!navigator.mediaDevices) {
      results.errors.push('MediaDevices API is not supported in your browser');
      return results;
    }
    results.mediaDevicesSupported = true;

    // Check getUserMedia support
    if (!navigator.mediaDevices.getUserMedia) {
      results.errors.push('getUserMedia is not supported in your browser');
      return results;
    }
    results.getUserMediaSupported = true;

    // Check enumerateDevices support
    if (!navigator.mediaDevices.enumerateDevices) {
      results.errors.push('enumerateDevices is not supported in your browser');
      return results;
    }
    results.enumerateDevicesSupported = true;

    return results;
  }

  static async checkCameraPermission() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach(track => track.stop());
      return { granted: true };
    } catch (error) {
      return {
        granted: false,
        error: error.name === 'NotAllowedError' ? 'Camera permission denied' :
               error.name === 'NotFoundError' ? 'No camera detected' :
               'Failed to access camera'
      };
    }
  }

  static async checkConnectedCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter(device => device.kind === 'videoinput');
      return {
        found: cameras.length > 0,
        count: cameras.length,
        devices: cameras
      };
    } catch (error) {
      return {
        found: false,
        count: 0,
        error: 'Failed to enumerate camera devices'
      };
    }
  }
}