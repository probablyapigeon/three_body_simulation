import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import miniaudio
import os
import math

def run_triple_body_simulation(matrix_size=64, audio_path=r"C:\Users\kiers\Downloads\He Can Try (Demo_Mix).mp3"):
    print(f"=== Initializing Patched Three-Body Phase Space Engine ===")
    
    use_audio_file = os.path.exists(audio_path)
    if use_audio_file:
        sound = miniaudio.decode_file(audio_path)
        sample_rate = sound.sample_rate
        data = np.array(sound.samples, dtype=np.float32)
        if sound.nchannels > 1: data = data.reshape(-1, sound.nchannels)[:, 0]
        audio_signal = data / np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else data
    else:
        sample_rate = 44100
        t_synth = np.linspace(0, 10, sample_rate * 10)
        audio_signal = np.sin(2 * np.pi * 60 * t_synth) * np.exp(-5 * (t_synth % 0.5))
        
    frames_per_second = 12
    total_frames = 150
    samples_per_frame = int(sample_rate / frames_per_second)

    # 1. FIXED: Enforce strict matrix symmetry at initialization
    X1_raw = torch.randn(matrix_size, matrix_size) * 0.05
    X2_raw = torch.randn(matrix_size, matrix_size) * 0.05
    X1 = (X1_raw + X1_raw.T) / 2.0
    X2 = (X2_raw + X2_raw.T) / 2.0
    
    P1 = torch.randn(matrix_size, matrix_size) * 0.02
    P2 = torch.randn(matrix_size, matrix_size) * 0.02
    
    dt_base = 0.03
    damping = 0.995  # Weak, single-pass dampening to retain orbital momentum
    
    body1_idx = 10
    body2_idx = 30
    body3_idx = 50

    fig = plt.figure(figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor('#05050A')
    ax = fig.add_subplot(111, projection='3d')

    trail1, trail2, trail3 = [], [], []

    def process_triple_body_step(frame_idx):
        nonlocal X1, X2, P1, P2
        if frame_idx >= total_frames: return
        ax.clear()
        ax.set_facecolor('#05050A')
        ax.grid(False)
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        
        # Audio feature extraction
        start, end = frame_idx * samples_per_frame, (frame_idx + 1) * samples_per_frame
        audio_chunk = audio_signal[start:end]
        if len(audio_chunk) < samples_per_frame: 
            audio_chunk = np.pad(audio_chunk, (0, samples_per_frame - len(audio_chunk)))
        fft_vals = np.abs(np.fft.rfft(audio_chunk * np.hanning(len(audio_chunk))))
        music_volt = float(min((np.mean(fft_vals) / np.max(fft_vals)) * 3.0, 3.0) if np.max(fft_vals) > 0 else 0)
        
        # 2. FIXED: Drive the global timeline step dynamically via audio voltage transients
        dt_eff = dt_base * (0.3 + music_volt * 1.5)
        
        omega_inner = frame_idx * 0.25
        omega_outer = frame_idx * 0.03
        
        defect_potentials = torch.zeros(matrix_size, matrix_size)
        defect_potentials[body1_idx, body1_idx] = float(3.0) * math.cos(omega_inner)
        defect_potentials[body2_idx, body2_idx] = float(2.0) * math.sin(omega_inner)
        defect_potentials[body3_idx, body3_idx] = float(1.2) * math.cos(omega_outer)
        
        # --- CLEAN VERLET LOOP (No intermediate dampening) ---
        X1_eff, X2_eff = X1 + defect_potentials, X2 + defect_potentials
        commutator = torch.mm(X1_eff, X2_eff) - torch.mm(X2_eff, X1_eff)
        F1 = -2.0 * (torch.mm(commutator, X2_eff) - torch.mm(X2_eff, commutator))
        F2 = 2.0 * (torch.mm(commutator, X1_eff) - torch.mm(X1_eff, commutator))
        
        F1 = torch.clamp(F1, min=-5.0, max=5.0)
        F2 = torch.clamp(F2, min=-5.0, max=5.0)
        
        # Velocity Half-Step
        P1 += F1 * (dt_eff / 2.0)
        P2 += F2 * (dt_eff / 2.0)
        
        # Position Full-Step
        X1 += P1 * dt_eff
        X2 += P2 * dt_eff
        
        X1_eff, X2_eff = X1 + defect_potentials, X2 + defect_potentials
        commutator_new = torch.mm(X1_eff, X2_eff) - torch.mm(X2_eff, X1_eff)
        F1_new = -2.0 * (torch.mm(commutator_new, X2_eff) - torch.mm(X2_eff, commutator_new))
        F2_new = 2.0 * (torch.mm(commutator_new, X1_eff) - torch.mm(X1_eff, commutator_new))
        
        F1_new = torch.clamp(F1_new, min=-5.0, max=5.0)
        F2_new = torch.clamp(F2_new, min=-5.0, max=5.0)
        
        # Velocity Final Half-Step
        P1 += F1_new * (dt_eff / 2.0)
        P2 += F2_new * (dt_eff / 2.0)
        
        # 3. FIXED: Apply weak velocity dampening strictly once per frame completion pass
        P1 *= damping
        P2 *= damping
        
        # Evaluate clean matrix potentials
        pe_val = -torch.trace(torch.mm(commutator_new, commutator_new)).item()
        ke_val = (torch.trace(torch.mm(P1, P1)) + torch.trace(torch.mm(P2, P2))).item()
        
        # Extract individual scalar trajectories
        xs = np.diag(X1.detach().numpy())
        ys = np.diag(X2.detach().numpy())
        zs = np.diag((P1 + P2).detach().numpy()) * 1.5
        
        b1 = (xs[body1_idx], ys[body1_idx], zs[body1_idx])
        b2 = (xs[body2_idx], ys[body2_idx], zs[body2_idx])
        b3 = (xs[body3_idx], ys[body3_idx], zs[body3_idx])
        
        trail1.append(b1); trail2.append(b2); trail3.append(b3)
        if len(trail1) > 45: trail1.pop(0); trail2.pop(0); trail3.pop(0)
        
        if len(trail1) > 1:
            t1, t2, t3 = np.array(trail1), np.array(trail2), np.array(trail3)
            ax.plot(t1[:, 0], t1[:, 1], t1[:, 2], color='#00FFFF', alpha=0.6, linewidth=1.2)
            ax.plot(t2[:, 0], t2[:, 1], t2[:, 2], color='#FF00FF', alpha=0.6, linewidth=1.0)
            ax.plot(t3[:, 0], t3[:, 1], t3[:, 2], color='#FFCC00', alpha=0.6, linewidth=0.8)
            
        ax.scatter([b1[0]], [b1[1]], [b1[2]], color='#00FFFF', s=150, edgecolors='#FFFFFF', linewidths=0.5)
        ax.scatter([b2[0]], [b2[1]], [b2[2]], color='#FF00FF', s=90, edgecolors='#FFFFFF', linewidths=0.5)
        ax.scatter([b3[0]], [b3[1]], [b3[2]], color='#FFCC00', s=110, edgecolors='#FFFFFF', linewidths=0.5)
        
        # Inter-body visual links
        ax.plot([b1[0], b2[0]], [b1[1], b2[1]], [b1[2], b2[2]], color='#FFFFFF', alpha=0.12, linewidth=0.8)
        cx, cy, cz = (b1[0]+b2[0])/2, (b1[1]+b2[1])/2, (b1[2]+b2[2])/2
        ax.plot([cx, b3[0]], [cy, b3[1]], [cz, b3[2]], color='#FFFFFF', alpha=0.06, linewidth=0.6)
        
        ax.view_init(elev=20, azim=frame_idx * 0.8)
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.4, 0.4)
        
        ax.set_title(
            f"Conserved Three-Body Phase Space Engine [PSR J0435+3233]\n"
            f"Clock Speed scale (dt_eff): {dt_eff:.4f} | Action PE: {pe_val:.2f}",
            color='#FFFFFF', fontsize=11, fontweight='bold', y=0.95
        )

    print("Compiling stabilized orbital phase space timelines...")
    ani = animation.FuncAnimation(fig, process_triple_body_step, frames=total_frames, interval=80, repeat=False)
    output_filename = "stable_three_body.gif"
    ani.save(output_filename, writer='pillow', fps=12)
    plt.close()
    print(f"Success! Orbital physics stabilized. File saved cleanly to: .\\{output_filename}")

if __name__ == "__main__":
    run_triple_body_simulation()
