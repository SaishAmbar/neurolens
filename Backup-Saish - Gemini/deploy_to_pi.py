"""Deploy camera_stream.py to Raspberry Pi via SSH (paramiko)."""
import paramiko
import time
import sys
import os

# Force UTF-8 output
os.environ["PYTHONIOENCODING"] = "utf-8"

PI_HOST = "10.165.194.112"
PI_USER = "pi"
PI_PASS = "pipipipi"
LOCAL_FILE = r"c:\Users\ambar.SAISH-ZEPHYRUS\OneDrive\Desktop\Backup-Srijan - Gemini\Backup-Saish - Gemini\camera_stream.py"
REMOTE_FILE = "/home/pi/camera_stream.py"
LOCAL_OLED = r"c:\Users\ambar.SAISH-ZEPHYRUS\OneDrive\Desktop\Backup-Srijan - Gemini\Backup-Saish - Gemini\oled_server.py"
REMOTE_OLED = "/home/pi/oled_server.py"

def run_cmd(ssh, cmd, timeout=10):
    print(f"  > {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out: print(f"    {out}")
    if err: print(f"    [stderr] {err}")
    return out, err

print("=" * 60)
print("  Deploying camera_stream.py to Pi")
print("=" * 60)

# Connect
print(f"\n[1/4] Connecting to {PI_USER}@{PI_HOST}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
print("  [OK] Connected!")

# Kill old process
print("\n[2/4] Killing any old camera_stream and oled_server processes...")
run_cmd(ssh, "pkill -f camera_stream.py || true")
run_cmd(ssh, "pkill -f oled_server.py || true")
run_cmd(ssh, "pkill -f 'python.*camera' || true")
run_cmd(ssh, "fuser -k -9 5001/tcp || true")
run_cmd(ssh, "fuser -k -9 5002/tcp || true")
time.sleep(3)
print("  [OK] Old processes killed, waiting for camera release...")

# Upload file
print(f"\n[3/4] Uploading camera_stream.py and oled_server.py...")
sftp = ssh.open_sftp()
sftp.put(LOCAL_FILE, REMOTE_FILE)
sftp.put(LOCAL_OLED, REMOTE_OLED)
sftp.close()
print(f"  [OK] Uploaded to {REMOTE_FILE} and {REMOTE_OLED}")

# Run the new script
print("\n[4/4] Starting oled_server.py and camera_stream.py on Pi...")
print("  (reading output for 20 seconds to confirm startup)\n")

transport = ssh.get_transport()
channel = transport.open_session()
# Start OLED server in background, then start camera stream in foreground
channel.exec_command(f"nohup python3 {REMOTE_OLED} > /dev/null 2>&1 & python3 {REMOTE_FILE} 2>&1")

# Read output for 20 seconds to confirm startup
start = time.time()
output = ""
while time.time() - start < 20:
    if channel.recv_ready():
        chunk = channel.recv(4096).decode(errors="replace")
        output += chunk
        # Print safely for Windows console
        safe = chunk.encode("ascii", errors="replace").decode()
        print(safe, end="", flush=True)
    time.sleep(0.3)

print("\n\n" + "=" * 60)
if "MJPEG" in output or "Camera started" in output or "Running on" in output:
    print("  SUCCESS! Camera stream is running!")
    print(f"  Stream : http://{PI_HOST}:5001/stream")
    print(f"  12MP   : http://{PI_HOST}:5001/capture")
    print(f"  HDR    : http://{PI_HOST}:5001/capture_hdr")
    print(f"  Health : http://{PI_HOST}:5001/health")
elif "Error" in output or "error" in output or "Traceback" in output:
    print("  ERROR - script failed to start. Output above.")
else:
    print("  Script started, no confirmation yet.")
    print(f"  Check: http://{PI_HOST}:5001/health")
print("=" * 60)
