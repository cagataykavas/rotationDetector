import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

# Configurable parameters
drop_wait_seconds = 1.1  # Seconds to wait after drop detection before analysis 
move_threshold     = 0.35  # Fraction threshold for movement detection per second
flow_mag_thresh    = 0.017  # Optical flow magnitude threshold per cell
debug_mode         = False  # New: if True, skip all plotting

# Convert optical flow to BGR color image
def flow_to_color(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    pause_each_sec = False
    file_index = 1

    script_dir = Path(__file__).parent
    # New: path for debug output
    debug_output_path = script_dir / "debug-output.txt"
    video_path = script_dir / "videolar" / "videolar" / f"test{file_index}.mp4"
    ref_path   = script_dir / f"referans{file_index}.txt"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read first frame for drop detection
    ret, first = cap.read()
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    # Drop detection parameters
    start_thresh, end_thresh = 0.2, 0.15
    drop_frame = None
    started = ended = False

    # Drop detection loop: show original + flow
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = cv2.cartToPolar(flow[...,0], flow[...,1])[0]
        avg_flow = mag.mean()

        combined = np.hstack((frame, flow_to_color(flow)))
        if not debug_mode:
            cv2.imshow('Detection + Flow', combined)
        prev_gray = gray

        # Detect start/end of drop
        if not started and avg_flow > start_thresh:
            started = True
        if started and not ended and avg_flow < end_thresh:
            ended = True
            drop_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            break
        if not debug_mode:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    # Load reference data if available
    if ref_path.exists():
        ref = []
        with open(ref_path, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split()
                ref.append(list(map(int, parts[1:10])))
        ref = np.array(ref[:60], int)
    else:
        ref = None

    # Compute homography at drop frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, drop_frame)
    ret, dframe = cap.read()
    h_frame, w_frame = dframe.shape[:2]
    gray0 = cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)
    dst = cv2.dilate(cv2.cornerHarris(cv2.blur(gray0,(5,5)).astype(np.float32),2,3,0.04), None)
    ys, xs = np.where(dst > 0.01 * dst.max())
    pts = np.stack((xs, ys), 1)
    s, d = xs+ys, xs-ys
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmax(d)], pts[np.argmin(d)]
    corners = np.array([tl, tr, br, bl], float)
    def dist(a, b): return np.linalg.norm(a - b)
    width = int(max(dist(corners[2], corners[3]), dist(corners[1], corners[0])))
    height = int(max(dist(corners[1], corners[2]), dist(corners[0], corners[3])))
    dst_pts = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], float)
    H, _ = cv2.findHomography(corners, dst_pts)

    # Grid and cell IDs
    cell_w, cell_h = width//3, height//3
    id_matrix = np.array([[7,1,4],[8,2,5],[9,3,6]])
    cell_ids = id_matrix.flatten()

    # Analysis after wait
    analysis_start = drop_frame + int(drop_wait_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, analysis_start)
    duration = 60
    fps_i = int(fps)
    counts_flow = np.zeros((9, duration), int)
    frame_cnt   = np.zeros(duration, int)
    prev_rect_gray = None
    f_idx = analysis_start

    # New: list to accumulate debug data per cell per frame
    debug_data = []  # each element: (frame_number, sec, cell index, mean_mag, mean_ang, count_mag, count_ang)

    while f_idx < analysis_start + duration * fps_i:
        ret, frame = cap.read()
        sec = (f_idx - analysis_start) // fps_i
        frame_cnt[sec] += 1

        # Rectify & grayscale
        rect = cv2.warpPerspective(frame, H, (width, height))
        gray_rect = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        rect_r = cv2.resize(rect, (w_frame, h_frame))

        # Blob detection (rover)
        quant = (gray_rect.astype(np.float32)/255 * 9).round().astype(np.uint8)
        mask = (quant == 0).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x,y,w_,h_ = cv2.boundingRect(c)
            if w_*h_ < 300: continue
            pad = int(min(w_,h_) * 0.1)
            x,y = max(0,x-pad), max(0,y-pad)
            boxes.append((x,y,w_+2*pad,h_+2*pad))
        sx, sy = w_frame/width, h_frame/height
        for x,y,w_,h_ in boxes:
            rx,ry = int(x*sx), int(y*sy)
            rw, rh = int(w_*sx), int(h_*sy)
            cv2.rectangle(rect_r, (rx,ry),(rx+rw, ry+rh),(255,0,0),2)

        # Draw cell IDs and ref values
        for idx in range(9):
            i,j = divmod(idx,3)
            cx, cy = j*cell_w, i*cell_h
            rx, ry = int(cx*sx)+5, int(cy*sy)+20
            cv2.putText(rect_r, f"R{cell_ids[idx]}", (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            if ref is not None:
                ref_val = ref[sec, cell_ids[idx]-1] if sec < len(ref) else -1
                text_ref = f"ref:{ref_val}"
            else:
                text_ref = ""
            cv2.putText(rect_r, text_ref, (rx, ry+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)

        # Optical flow detection
        flow_vis = np.zeros_like(rect_r)
        if prev_rect_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_rect_gray, gray_rect, None,0.5,3,15,3,5,1.2,0)
            mag_map, ang_map = cv2.cartToPolar(flow[...,0], flow[...,1])
            flow_vis = cv2.resize(flow_to_color(flow), (w_frame, h_frame))

            # Per-cell magnitude detection with debug logging
            for idx in range(9):
                i, j = divmod(idx, 3)
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell_height = y2 - y1
                cell_width  = x2 - x1
                ys_coords, xs_coords = np.mgrid[0:cell_height, 0:cell_width]
                rx_offset = xs_coords - cell_width / 2
                ry_offset = ys_coords - cell_height / 2
                vx = flow[y1:y2, x1:x2, 0]
                vy = flow[y1:y2, x1:x2, 1]
                mean_mag = np.mean(mag_map[y1:y2, x1:x2])
                mean_ang = np.mean(ang_map[y1:y2, x1:x2])
                count_mag = np.count_nonzero(mag_map[y1:y2, x1:x2] >= flow_mag_thresh)
                # Hareket algılandıysa yeşil, değilse kırmızı dikdörtgen
                is_moving = mean_mag >= flow_mag_thresh
                if is_moving:
                    counts_flow[idx, sec] += 1
                color = (0, 255, 0) if is_moving else (0, 0, 255)
                rx1, ry1 = int(x1 * sx), int(y1 * sy)
                cv2.rectangle(rect_r, (rx1, ry1), (int(x2 * sx), int(y2 * sy)), color, 2)
                text = f"m{mean_mag:.3f}"
                cv2.putText(rect_r, text, (rx1 + 5, ry1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # New: append debug info for this cell and frame
                debug_data.append((f_idx, sec, cell_ids[idx], mean_mag, mean_ang, count_mag, 0))
        prev_rect_gray = gray_rect

        # Show result
        combined = np.hstack((rect_r, flow_vis))
        if not debug_mode:
            cv2.putText(combined, f"Sec:{sec+1} Drop@{drop_frame/fps:.2f}s", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            cv2.imshow('Analysis', combined)
            if pause_each_sec:
                cv2.waitKey(1)
            elif cv2.waitKey(1)&0xFF==ord('q'):
                break
        f_idx += 1

    # Write debug output (mean_mag, mean_ang, count_mag, count_ang for each cell per frame)
    with open(debug_output_path, 'w') as dbg:
        dbg.write("Frame\tSec\tCell\tMean_Mag\tMean_Ang\tCount_Mag\tCount_Ang\n")
        for rec in debug_data:
            dbg.write("\t".join(str(x) for x in rec) + "\n")

    cap.release()
    if not debug_mode:
        cv2.destroyAllWindows()

    # build movement matrix
    movement = np.zeros((duration,9), int)
    threshold_frames = int(np.ceil(move_threshold * fps_i))  # örn. 0.2 * 30fps → 6 kare
    threshold_frames = threshold_frames + 0
    for s in range(duration):
        if frame_cnt[s] > 0:
            for idx in range(9):
                if counts_flow[idx, s] >= threshold_frames:
                    movement[s, idx] = 1

    # reorder to Robot-1…Robot-9
    order = [list(cell_ids).index(r) for r in range(1,10)]
    mov_ord = movement[:, order]

    # write to file
    out_file = script_dir / f"ogr.txt"
    with open(out_file, 'w') as f:
        f.write('Saniye\t' + '\t'.join(f'Robot-{i}' for i in range(1,10)) + '\n')
        for s, row in enumerate(mov_ord):
            f.write(f" {s+1})\t" + '\t'.join(str(v) for v in row) + '\n')
    print(f"Output written to {out_file} and debug info to {debug_output_path}")
    
    # New: Write per-cell detailed and summary analysis outputs into a new folder "cell_logs"
    cell_folder = script_dir / "cell_logs"
    os.makedirs(cell_folder, exist_ok=True)
    # Build a mapping from cell number to its index in cell_ids
    cell_map = {cell: cell_ids.tolist().index(cell) for cell in range(1,10)}
    # Dictionary to accumulate per-cell per-frame debug data from debug_data list
    cell_details = {cell: [] for cell in range(1,10)}
    for rec in debug_data:
        # rec = (frame, sec, cell, mean_mag, mean_ang, count_mag, count_ang)
        cell_details[rec[2]].append(rec)
    
    # Write per-frame detailed log and per-second summary files for each cell
    for cell in range(1,10):
        # Detailed per-frame log for cell X
        detail_path = cell_folder / f"cell{cell}.txt"
        with open(detail_path, 'w') as fcell:
            fcell.write("Frame\tSec\tMean_Mag\tMean_Ang\tCount_Mag\tCount_Ang\n")
            for rec in cell_details[cell]:
                # Write: frame, sec, mean_mag, mean_ang, count_mag, count_ang
                fcell.write("\t".join(str(x) for x in (rec[0], rec[1], rec[3], rec[4], rec[5], rec[6])) + "\n")
    
        # Summary analysis per second using counts_flow from the main loop
        summary_path = cell_folder / f"cell{cell}analysis.txt"
        with open(summary_path, 'w') as fcell:
            fcell.write("Sec\tCount_Threshold_Passed\n")
            idx = cell_map[cell]
            for s in range(duration):
                fcell.write(f"{s+1}\t{counts_flow[idx, s]}\n")

    # New: Plot and write Cell 1 magnitude values
    # Filter out debug data for Cell 1; rec = (frame, sec, cell, mean_mag, mean_ang, count_mag, count_ang)
    cell1_data = [rec for rec in debug_data if rec[2] == 1]
    frames_cell1 = [rec[0] for rec in cell1_data]
    mean_mags = [rec[3] for rec in cell1_data]
    
    plt.figure(figsize=(10,5))
    plt.plot(frames_cell1, mean_mags, label="Cell 1 Mean Magnitude")
    plt.axhline(flow_mag_thresh, color="red", linestyle="--", label="Magnitude Threshold")
    # Draw vertical lines for each second boundary
    for n in range(duration):
         frame_line = analysis_start + n * fps_i
         plt.axvline(frame_line, color="gray", linestyle=":", alpha=0.3)
         plt.text(frame_line, flow_mag_thresh+0.05, f"{n+1}s", rotation=90, verticalalignment='bottom', fontsize=8)
    
    plt.xlabel("Frame")
    plt.ylabel("Mean Magnitude")
    plt.title("Cell 1 Mean Magnitude vs Frame")
    plt.legend()
    plt.tight_layout()
    plt.savefig(script_dir / "cell1_magnitude_plot.png")
    if not debug_mode:
         plt.show()

if __name__ == '__main__':
    main()
