### FUSION SCRIPT

import adsk.core
import adsk.fusion
import traceback
import socket
import math

# --- CONSTANTS ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
BUFFER_SIZE = 1024
TRANSLATION_SCALE = 30.0
BASE_VIEW_EXTENTS = 150.0
ORBIT_RADIUS = 50.0


class HandControllerSession:
    """Manages the lifecycle and camera logic for the Hand Controller session."""

    def __init__(self):
        self.app = adsk.core.Application.get()
        self.ui = self.app.userInterface
        self.viewport = self.app.activeViewport
        self.socket = self._setup_socket()

    def _setup_socket(self):
        """Initializes a non-blocking UDP socket with address reuse."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((UDP_IP, UDP_PORT))
        sock.setblocking(False)
        return sock

    def update_camera(self, pan_x, pan_y, zoom, pitch, yaw):
        """Calculates and applies new camera transforms based on telemetry."""
        # Calculate target pivot (Translation)
        target = adsk.core.Point3D.create(pan_x * TRANSLATION_SCALE, pan_y * TRANSLATION_SCALE, 0)

        # Calculate eye position (Spherical Orbit)
        # pitch: vertical elevation, yaw: horizontal rotation
        eye_x = target.x + ORBIT_RADIUS * math.cos(pitch) * math.sin(yaw)
        eye_y = target.y - ORBIT_RADIUS * math.cos(pitch) * math.cos(yaw)
        eye_z = target.z + ORBIT_RADIUS * math.sin(pitch)
        eye = adsk.core.Point3D.create(eye_x, eye_y, eye_z)

        # Apply transforms to camera object
        camera = self.viewport.camera
        camera.isSmoothTransition = False
        camera.target = target
        camera.eye = eye
        camera.upVector = adsk.core.Vector3D.create(0, 0, 1)  # Lock Z-axis

        # Apply Zoom level via View Extents
        # Normalized zoom factor scaled for CAD visibility
        effective_zoom = zoom * 2.0
        camera.viewExtents = max(1.0, BASE_VIEW_EXTENTS / max(0.1, effective_zoom))

        self.viewport.camera = camera
        self.viewport.refresh()

    def run(self):
        """Main execution loop."""
        self.ui.messageBox("Session Started\nClick 'Cancel' or Esc to terminate controller link.")

        try:
            while True:
                # Pump UI events to keep Fusion responsive and allow termination
                adsk.doEvents()

                # Check for termination signal (Command end or App close)
                if not self.ui.activeCommand:
                    break

                try:
                    data, _ = self.socket.recvfrom(BUFFER_SIZE)
                    packet = data.decode().split(',')
                    if len(packet) < 5: continue

                    # Unpack telemetry data
                    pan_x, pan_y, zoom, pitch, yaw = map(float, packet[:5])
                    self.update_camera(pan_x, pan_y, zoom, pitch, yaw)

                except (BlockingIOError, socket.error):
                    continue
                except Exception as e:
                    print(f"Data processing error: {e}")
                    continue

        finally:
            self.socket.close()


def run(context):
    session = None
    try:
        session = HandControllerSession()
        session.run()
    except:
        app = adsk.core.Application.get()
        if app.userInterface:
            app.userInterface.messageBox(f'Controller session terminated unexpectedly:\n{traceback.format_exc()}')