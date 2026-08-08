
import cv2
import numpy as np
import matlab.engine

#CONSTANTS --------------------------------------------------------------------------------
WINDOW_NAME  = "FrequencyCam - Real-time FFT Filter"
WEBCAM_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
DISPLAY_SIZE = 320

#FILTER MODES --------------------------------------------------------------------------------
FILTER_MODES = {
    ord('l') : 'lowpass',
    ord('h') : 'highpass',
    ord('s') : 'sharpen',
    ord('d') : 'none', #try karke dekhte
    ord('m') : 'homomorphic',
}

#Labels shown on screen for each mode.
MODE_LABELS = {
    'lowpass'  : 'L low-pass  (Blur)',
    'highpass' : 'H high-pass (Edges)',
    'sharpen'  : 'S sharpen   (Boost edges)',
    'none'     : 'No filter   (Raw webcam)',
    'homomorphic' : 'M Homomorphic (Light fix)',
}

#TRACKBAR NAMES --------------------------------------------------------------
TB_RADIUS    = 'FiltRadius'
TB_DC        = 'DC Protect'
TB_INTENSITY = 'Intensity'

#STARTUP ---------------------------------------------------------------------
def start_matlab_engine():
    print("[INFO] Starting MATLAB engine - this takes ~10 seconds...")
    engine = matlab.engine.start_matlab()                               #engine is the object returned. matlab.engine.start_matlab() starts a MATLAB sesh in the background
    print("[INFO] MATLAB engine ready")
    return engine

def open_webcam():
    webcam = cv2.VideoCapture(WEBCAM_INDEX)                              #starts webcam
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)                    #fix the width of the webcam to 640
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)                  #fix the height of the webcam to 640

    if not webcam.isOpened():
        raise RuntimeError("[ERROR] Could not open webcam. Check WEBCAM_INDEX.")
    
    print("[INFO] Webcam opened successfully.")
    return webcam

def setup_window():
    cv2.namedWindow(WINDOW_NAME)                                             #creates a name popup window

    #draw the sliders on the window
    # trackbars       (label,        window,    default, max, callback). We use"lambda x : None" cuz
    cv2.createTrackbar(TB_RADIUS,    WINDOW_NAME, 32, 150, lambda x: None)
    cv2.createTrackbar(TB_DC,        WINDOW_NAME, 15,  60, lambda x: None)
    cv2.createTrackbar(TB_INTENSITY, WINDOW_NAME, 80, 100, lambda x: None)

    print("[INFO] Window and trackbars ready!")
    print("[INFO] Keysc : L=LowPass  H=HighPass  S=Sharpen M=Homomorphic  SPACE=Snapshot  Q=Quit")

#READ TRACKBAR VALUES EACH FRAME ---------------------------------------------
def read_trackbars():
    filter_radius     = cv2.getTrackbarPos(TB_RADIUS,    WINDOW_NAME)
    dc_protect_radius = cv2.getTrackbarPos(TB_DC,        WINDOW_NAME)
    intensity         = cv2.getTrackbarPos(TB_INTENSITY, WINDOW_NAME) / 100.0      #intensity trackbar goes from 1to100, but MATLAB needs 0to1. Hence, /100.0

    filter_radius     = max(1, filter_radius)     #To ensure we dont get 0
    dc_protect_radius = max(1, dc_protect_radius)

    return filter_radius, dc_protect_radius, intensity

#SEND FRAME TO MATLAB, GET BACK THE RESULT ---------------------------------------------------------
def run_matlab_filter(engine, hsv_frame, filter_mode, filter_radius, dc_protect_radius, intensity):
    v_channel    = hsv_frame[:,:,2]                                                                 # We extract V (brightness channels only; V is the 2nd array of the 3 arrays of HSV)
    matlab_frame = matlab.uint8(v_channel.tolist())                                                 #Convert numpy array -> matlab-compatible. matlab.uint8 needs a list of rows.

                                                                                                    #Call fft_filter4.m nargout=1 here because this will return in total 1 output
    filtered_matlab = engine.fft_filter5(
        matlab_frame, 
        filter_mode,
        float(filter_radius),
        float(dc_protect_radius),
        float(intensity),
        nargout=1
    )

    filtered_v = np.array(filtered_matlab, dtype=np.uint8)                  #MATLAB returns the final array. This function converts it back to np array
    hsv_result = hsv_frame.copy()                                           #BGR result is started from a copy of HSV; so that we dont alter the OG image being displayed
    hsv_result[:,:,2] = filtered_v                                          #put back filtered V; H&S are untouched.
    bgr_result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)                #convers HSV image to BGR

    return bgr_result
    

#BUILD MAGNITUDE SPECTRUM DISPLAY IMAGE --------------------------------------------------------------------------------
def compute_magnitude_display(gray_frame):
    freq_domain   = np.fft.fft2(gray_frame)             #fft2 transform applied by MATLAB to go from spatial -> frequency domain
    freq_shifted  = np.fft.fftshift(freq_domain)        #fftshift moves the bright DC centre (main object) to the centre
    magnitude     = np.abs(freq_shifted)                #gets the magnitude of the complex nos (we take magnitude cuz we display "How much?" of the frequency exists in the image)
    magnitude_log = np.log1p(magnitude)                 #logscale computes log(1+input) so that log(0) is avoided and peaks are visibile (shrinks down a large range to a small range)

    
    magnitude_norm  = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)   #normalize the entire range in 0to255
                                                                                    #inputs are (array, destination, alpha, beta, normalisation type)
                                                                                    #alpha & beta are start & end of output range. .NORM_MINMAX stretches as alpha=lowest & beta=biggest
    magnitude_uint8 = np.uint8(magnitude_norm)

    #Apply a colormap so it looks distinct from the gray panels
    magnitude_colored = cv2.applyColorMap(magnitude_uint8, cv2.COLORMAP_MAGMA)

    return magnitude_colored

#DRAW TEXT OVERLAYS ON A PANEL --------------------------------------------------------------------------------
def draw_label(panel, text, position=(10,22), color=(0,255,0)):
    cv2.putText(
        panel, text, position, 
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA
    )

#ASSEMBLE THE 2X2 DISPLAY GRID --------------------------------------------------------------------------------
def build_display_grid(bgr_frame, magnitude_colored, filtered_frame, active_mode, fps):
    size = (DISPLAY_SIZE, DISPLAY_SIZE)                                                                         #320x320

    #Panel1 : OG webcam
    panel_original = cv2.resize(bgr_frame, size)                                                                #resize the output frame to desired dimensions
    draw_label(panel_original, "ORIGINAL")
    draw_label(panel_original, f"f{fps:.1f} fps", position=(10, DISPLAY_SIZE-10), color=(180, 180, 180))

    #Panel2 : FFT magnitude spectrum
    panel_magnitude = cv2.resize(magnitude_colored, size)
    draw_label(panel_magnitude, "FFT MAGNITUDE", color=(200, 200, 255))

    #Panel3 : Active Mode label
    panel_mode = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 3), dtype=np.uint8)                                      #creates a black array. This is the BG for the modes display
    draw_label(panel_mode, "ACTIVE MODE", position=(10, 22), color=(180, 180, 180))
    draw_label(panel_mode, MODE_LABELS.get(active_mode, active_mode), position=(10, 55), color=(100, 255, 150))
    draw_label(panel_mode, "L/H/N/S to switch", position=(10, DISPLAY_SIZE-10), color=(100, 100, 100))

    #Panel4 : Filtered output
    panel_filtered = cv2.resize(filtered_frame, size)
    draw_label(panel_filtered, "FILTERED OUTPUT", color=(255, 200, 100))

    #Stich the 2x2 grid
    top_row   = np.hstack([panel_original, panel_magnitude])  #OG & FFT on the top 2
    bottom_row= np.hstack([panel_mode, panel_filtered])       #mode display & filtered on the botton 2
    full_grid = np.vstack([top_row, bottom_row])              #stack them vertically

    return full_grid

 
#SAVE SNAPSHOT --------------------------------------------------------------------------------
def save_snapshot(display_grid, snapshot_count):
    filename=f"snapshot_{snapshot_count:03d}.png"
    cv2.imwrite(filename, display_grid)
    print(f"[INFO] Snapshot saved -> {filename}")


#MAIN LOOP --------------------------------------------------------------------------------
def main():
    matlab_engine = start_matlab_engine()

    #CHANGE PATH HERE IS ANOTHER FOLDER CREATED!!
    matlab_engine.addpath(r'C:\Users\kadam\Desktop\SnapFFT\final', nargout=0)  #tells matlab where fft_filter4.m is kept. nargout=0 means this function wont return anything; so dont stall

    webcam = open_webcam()
    setup_window()

    active_mode    = 'none'  #defautl filter on startup
    snapshot_count = 0
    prev_tick      = cv2.getTickCount()

    try:
        while True:

            #Grab frame
            frame_grabbed, bgr_frame = webcam.read()
            if not frame_grabbed:
                print("[WARN] Dropped frame - skipping.")
                continue
            
            #Convert to grayscale
            hsv_frame=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)           #converts BGR image frame to HSV

            #Read slider values
            filter_radius, dc_protect_radius, intensity = read_trackbars()

            #Send to MATLAB, get filtered frame back
            filtered_frame = run_matlab_filter(matlab_engine, hsv_frame, active_mode, filter_radius, dc_protect_radius, intensity)

            #Compute FFT magnitude for display
            magnitude_display = compute_magnitude_display(hsv_frame[:,:,2])

            #Calculate FPS
            current_tick = cv2.getTickCount()
            fps          = cv2.getTickFrequency() / (current_tick - prev_tick)
            prev_tick    = current_tick

            #Build and show the 2x2 grid
            display_grid = build_display_grid(bgr_frame, magnitude_display, filtered_frame, active_mode, fps)
            cv2.imshow(WINDOW_NAME, display_grid) 

            #Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key==ord('q'):
                print("[INFO] Q pressed - Quitting.")
                break
            
            elif key == ord(' '):
                snapshot_count += 1
                save_snapshot(display_grid, snapshot_count)

            elif key in FILTER_MODES:
                active_mode=FILTER_MODES[key]
                print(f"[INFO] Filter switched -> {active_mode}")

    finally:
        print("[INFO] Releasing webcam and closing windows")
        webcam.release()
        cv2.destroyAllWindows()
        matlab_engine.quit()                #shuts down MATLAB sesh cleanly
        print("[INFO] Done!")


#ENTRY POINT
if __name__ == "__main__":
    main()



