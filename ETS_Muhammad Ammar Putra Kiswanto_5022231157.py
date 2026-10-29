from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# ==========================================
# 1. KONEKSI & SETUP
# ==========================================
print("Menghubungkan ke CoppeliaSim...")
client = RemoteAPIClient()
sim = client.require('sim')
sim.setStepping(True)  # Mode sinkron aktif agar gerakan presisi

drone = sim.getObject('/Quadcopter')
target = sim.getObject('/Quadcopter/target')

# FIX UTAMA: Menunjuk langsung ke "sensor" di dalam fast3DLaserScanner
cam = sim.getObject('/Quadcopter/fast3DLaserScanner/sensor')

# Mengambil parameter intrinsik sensor
near = sim.getObjectFloatParam(cam, sim.visionfloatparam_near_clipping)
far = sim.getObjectFloatParam(cam, sim.visionfloatparam_far_clipping)
fov = sim.getObjectFloatParam(cam, sim.visionfloatparam_perspective_angle)
res = sim.getVisionSensorRes(cam)
W, H = res[0], res[1]
f = (max(W, H) / 2.0) / np.tan(fov / 2.0)

us, vs = np.meshgrid(np.arange(W), np.arange(H))
X_RAY = -(us - (W - 1) / 2.0) / f
Y_RAY = (vs - (H - 1) / 2.0) / f

# ==========================================
# 2. PARAMETER JALUR (LAWNMOWER) & TIMER
# ==========================================
ALT = 1.5
Y_RANGE = np.arange(-2.0, 2.5, 1.0)
X_RANGE = [-2.0, 2.0]

waypoints = []
for i, y in enumerate(Y_RANGE):
    if i % 2 == 0:
        waypoints.extend([[X_RANGE[0], y, ALT], [X_RANGE[1], y, ALT]])
    else:
        waypoints.extend([[X_RANGE[1], y, ALT], [X_RANGE[0], y, ALT]])

SPEED = 0.5
DURASI_MAKSIMAL = 30.0  # Simulasi berhenti otomatis setelah 30 detik
WP_TOLERANSI = 0.15

cloud_chunks = []
drone_path = []
wp_i = 0

# ==========================================
# 3. KONTROL & MAPPING (MAIN LOOP)
# ==========================================
sim.startSimulation()
dt = sim.getSimulationTimeStep()
tgt_pos = np.array(sim.getObjectPosition(target, sim.handle_world))

print(f"Simulasi berjalan. Batas waktu: {DURASI_MAKSIMAL} detik.")

try:
    while wp_i < len(waypoints):
        if sim.getSimulationTime() > DURASI_MAKSIMAL:
            print("\nWaktu habis. Berhenti mapping.")
            break

        wp = waypoints[wp_i]
        diff = np.array(wp) - tgt_pos
        dist = np.linalg.norm(diff)
        
        if dist > 1e-6:
            tgt_pos = tgt_pos + (diff / dist) * min(SPEED * dt, dist)
        sim.setObjectPosition(target, sim.handle_world, tgt_pos.tolist())

        pos = sim.getObjectPosition(drone, sim.handle_world)
        
        if np.linalg.norm(np.array(pos) - np.array(wp)) < WP_TOLERANSI:
            wp_i += 1

        # Rekam titik jalur dan data sensor secara sinkron
        step_count = int(sim.getSimulationTime() / dt)
        if step_count % 3 == 0:
            drone_path.append(pos)
            
            buf, _ = sim.getVisionSensorDepth(cam, 1)
            d = np.frombuffer(buf, dtype=np.float32).reshape(H, W).astype(np.float64)
            
            if d.max() > 1.0001:
                z = d
            else:
                z = near + d * (far - near)
                
            valid = (z < far - 0.05) & (z > near + 0.02)
            P = np.stack([X_RAY * z, Y_RAY * z, z], axis=-1)[valid]
            
            if len(P) > 0:
                m = sim.getObjectMatrix(cam, sim.handle_world)
                M = np.eye(4)
                M[:3, :] = np.array(m).reshape(3, 4)
                Pw = (M[:3, :3] @ P.T).T + M[:3, 3]
                cloud_chunks.append(Pw)

        sim.step() 
finally:
    sim.stopSimulation()
    print("Memproses visualisasi 3D...")

# ==========================================
# 4. POST-PROCESSING & PLOTTING
# ==========================================
if len(cloud_chunks) > 0:
    raw = np.vstack(cloud_chunks)
    drone_path_np = np.array(drone_path)

    # Voxel filtering via Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw)
    downpcd = pcd.voxel_down_sample(voxel_size=0.08)
    vox = np.asarray(downpcd.points)

    print(f"Data Raw: {len(raw)} -> Voxel: {len(vox)}")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotting Point Cloud
    sc = ax.scatter(vox[:, 0], vox[:, 1], vox[:, 2], c=vox[:, 2], cmap='RdYlGn_r', s=4, alpha=0.8)
    
    # Plotting Jalur Drone
    if len(drone_path_np) > 0:
        ax.plot(drone_path_np[:, 0], drone_path_np[:, 1], drone_path_np[:, 2], color='blue', lw=3, label='Jalur drone')
        ax.scatter(*drone_path_np[0], color='black', s=60, marker='o', label='Start')
        ax.scatter(*drone_path_np[-1], color='magenta', s=60, marker='X', label='Finish')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Map - Hasil UTS')
    
    # Penyesuaian Sudut Pandang dan Proporsi Kotak (Sudah diperbaiki tanda kurungnya)
    zmax = max(vox[:, 2].max(), drone_path_np[:, 2].max()) if len(drone_path_np) > 0 else vox[:, 2].max()
    ax.set_zlim(0, zmax * 1.05)
    ax.view_init(elev=35, azim=-60)
    fig.colorbar(sc, label='Ketinggian (m)', shrink=0.6)
    
    ax.legend()
    plt.show()
else:
    print("Gagal mengambil data. Pastikan ada objek di dekat drone (area scan valid).")