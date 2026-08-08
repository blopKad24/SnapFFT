%Function is given a single grayscale image frame from the webcam,
%we tell it which filter to apply and at what strength
%and it returns a cleaned/processed version of that frame. 

%Internally it converts the image into the frequency domain using FFT (where each point represents how much of a certain pattern exists), 
%builds a mask that either blocks or boosts specific frequencies depending on the chosen filter, 
%multiplies the FFT by that mask; then converts back to a real image using IFFT. 
%Finally it blends the result with the original based on the intensity slider, clips values to valid pixel range and returns an uint8 image (displayed by OpenCV).

% FFT_FILTER  Apply frequency-filter to a single webcam frame. 
%   INPUTS:
%     raw_frame         - grayscale image matrix (uint8) from Python/webcam
%     filter_mode       - string: 'lowpass', 'highpass', 'homomorphic', 'sharpen'
%     filter_radius     - radius in pixels controlling filter cutoff
%     dc_protect_radius - radius around center (DC) to always preserve
%     intensity         - blend factor 0.0 to 1.0 (how strong the filter is)
%
%   OUTPUT:
%     filtered_frame    - processed grayscale image (uint8), same size as input

function filtered_frame=fft_filter5(raw_frame, filter_mode, filter_radius, dc_protect_radius, intensity)  %python uses this function for every frame of the webcam
                                                                                                          %applies frequency filter to the image

                                                        %STEP1a.Prepare the frame.
  gray_double=double(raw_frame);                        %raw frame is 0to255 uint8. FFT needs decimal. Thus double() used
  [frame_height, frame_width]=size(gray_double);

                                                        %STEP1b.Log transform
  if strcmpi(filter_mode, 'homomorphic')
      work_frame = log1p(gray_double);                  %log1p(x) does log(1+x); which avoids log(0)
  else
      work_frame = gray_double;                         %only homomorphic filter needs log; rest work on the raw gray pixels
  end

                                                        %STEP2 : FFT
  freq_domain =fft2(work_frame);                        %fft2 converts spatial -> frequency domain
  freq_shifted=fftshift(freq_domain);                   %fftshift brings DC centre (main object) to the centre of the array


                                                        %STEP3 : Build & apply the Frequency mask
  freq_mask = build_freq_mask(frame_height, frame_width, filter_mode, filter_radius, dc_protect_radius);

                                                        %CORE OF THE
                                                        %PROJECT: One
                                                        %multiplication in
                                                        %freq domain that
                                                        %would take many
                                                        %edits in the
                                                        %spatial
  freq_filtered = freq_shifted .* freq_mask;            % 0 = block the frequency
                                                        % 1 = keep the frequency
                                                        %>1 = boost (sharpen)


                                                        %STEP4 : IFFT
  freq_unshifted =ifftshift(freq_filtered);
  spatial_result =real(ifft2(freq_unshifted));          %real() needed to remove the garbage imaginary part (due to floating point arithmetic)


                                                        %STEP5 : Blend with the OG
  if strcmpi(filter_mode, 'homomorphic')
      spatial_result = expm1(spatial_result);           %undo the log — expm1 is exp(x)-1
      spatial_result = max(0, spatial_result);          %clamp negatives from floating point
  end
  blended = intensity*spatial_result + (1-intensity)*gray_double;  %if intensity is 80, then 0.8*filtered + 0.2*OG


                                                        %STEP6 : Clip & convert back to uint8
  blended        =max(0, min(255, blended));            %safety measure to ridden any floating point arithmetic roudning errors
  filtered_frame =uint8(blended);                       %OpenCV displays this uint8 filtered image (values 0to255)

end

% -------------------------------------------------------------------------


function freq_mask = build_freq_mask(frame_height, frame_width, filter_mode, filter_radius, dc_protect_radius)
  
                                                    % BUILD_FREQ_MASK  Create a 2D mask to keep or block frequency regions.
                                                    %the mask is 1 = keep this frequency, 0 = block this frequency.
                                                    %all filters use a smooth Gaussian edge (no hard cutoffs) to avoid
                                                    %ringing artifacts in the reconstructed image.

                                                 %distance map from centre:
                                                 %every pixel in the frequency domain gets a distance value; telling us how far it is from the zero-frequency DC centre.
                                                 %(needed for Lowpass & Highpass)
  center_row = floor(frame_height / 2) + 1;      % +1 as MATLAB starts arrays with 1
  center_col = floor(frame_width  / 2) + 1;

  col_indices = 1:frame_width;
  row_indices = (1:frame_height)';

  dist_from_center = sqrt((col_indices - center_col).^2 + (row_indices - center_row).^2);

  switch lower(filter_mode)

      case 'lowpass'                                                        %pass low freqs(centre), block high freqs(edges). 
          freq_mask  = gaussian_rolloff(dist_from_center, filter_radius); 
      
      case 'highpass'                                                       %block low freqs (centre), pass high freqs (edges). 
          low_freq_part = gaussian_rolloff(dist_from_center, filter_radius);
          freq_mask     = 1-low_freq_part;                                  %now fades from 0 -> 1 (block low near DC, pass high near edges)

          dc_protection = gaussian_rolloff(dist_from_center, dc_protect_radius); %the circular region that we protect(Main obj in the frame)
          freq_mask     = freq_mask + dc_protection;
          freq_mask     = min(1, freq_mask);                                %safety measure to avoid >1 values

      case 'sharpen'                                                        %boost the high freqs.
          high_freq_boost = 1-gaussian_rolloff(dist_from_center, filter_radius);
          freq_mask       = 1+high_freq_boost;

     case 'homomorphic'                                                     %boosts frequency after log transform.
          high_freq_part = 1-gaussian_rolloff(dist_from_center, filter_radius);
          dc_protection  = gaussian_rolloff(dist_from_center, dc_protect_radius);
          freq_mask      = 0.5 +  1.5.*high_freq_part;                            %range is 0.5 to 2.0 (low to high).
          freq_mask      = freq_mask .*(1-dc_protection) + dc_protection;         %protect DC part.

      otherwise                                                              %unknown mode. Return a flat mask (No change in image).
          freq_mask = ones(frame_height, frame_width);
  end
end

% -------------------------------------------------------------------------


function smooth_mask = gaussian_rolloff(dist_from_center, cutoff_radius)   %smooth circular mask that fades from 1 -> 0 at cutoff_radius.
                                                                           %This is how low freqs remain and high freqs are blocked. For Highpass, this filter is simply inverted

  smooth_mask =exp(-(dist_from_center .^ 2) / (2 * cutoff_radius ^ 2));    %Gaussian Bell curve formula

end

% -------------------------------------------------------------------------
