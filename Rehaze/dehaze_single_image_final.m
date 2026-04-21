function J = dehaze_single_image(inputPath, outputPath)

    % Read input image and convert to double precision
    I = im2double(imread(inputPath));

    % Ensure image is in RGB format
    if size(I, 3) == 1
        I = repmat(I, [1 1 3]);
    end

    % Parameter initialization
    omega = 0.9;        
    t0 = 0.2;           
    patchSize = 15;

    % Compute dark channel of the image
    dark = getDarkChannel(I, patchSize);

    % Estimate atmospheric light
    A = estimateAtmosphere(I, dark);

    % Normalize atmospheric light to reduce color distortion
    A = mean(A) * [1 1 1];

    % Normalize image with respect to atmospheric light
    normI = zeros(size(I));
    for c = 1:3
        normI(:, :, c) = I(:, :, c) / max(A(c), 1e-6);
    end

    % Estimate transmission map
    transmission = 1 - omega * getDarkChannel(normI, patchSize);

    % Refine transmission map using filtering
    transmission = refineTransmission(I, transmission);

    % Apply lower bound to transmission
    transmission = max(transmission, t0);

    % Recover scene radiance
    J = zeros(size(I));
    for c = 1:3
        J(:, :, c) = (I(:, :, c) - A(c)) ./ transmission + A(c);
    end

    % Clip output values to valid range
    J = min(max(J, 0), 1);

    % Normalize brightness of output image
    J = J ./ max(J(:));

    % Apply gamma correction for visual enhancement
    J = imadjust(J, [], [], 0.9);

    % Apply sharpening filter to enhance edges
    J = imsharpen(J, 'Radius', 1, 'Amount', 0.8);

    % Save output image if path is provided
    if nargin > 1 && ~isempty(outputPath)
        imwrite(J, outputPath);
    end

    % Display input and output images
    figure('Name', 'Improved Dehazing Result');
    subplot(1,2,1); imshow(I); title('Input');
    subplot(1,2,2); imshow(J); title('Output');
end


function dark = getDarkChannel(I, patchSize)
    % Compute minimum intensity across RGB channels
    minChannel = min(I, [], 3);

    % Apply morphological erosion to obtain dark channel
    se = strel('square', patchSize);
    dark = imerode(minChannel, se);
end


function A = estimateAtmosphere(I, dark)
    % Determine image size
    [h, w, ~] = size(I);
    numPixels = h * w;

    % Select top 0.1% brightest pixels in dark channel
    numBright = max(round(numPixels * 0.001), 1);

    % Reshape data for processing
    darkVec = dark(:);
    imageVec = reshape(I, numPixels, 3);

    % Sort pixels based on dark channel intensity
    [~, indices] = sort(darkVec, 'descend');
    topIdx = indices(1:numBright);

    % Extract corresponding brightest pixels
    brightest = imageVec(topIdx, :);

    % Select pixel with maximum intensity sum
    [~, maxIdx] = max(sum(brightest, 2));
    A = brightest(maxIdx, :);
end


function tRefined = refineTransmission(I, t)
    % Convert image to grayscale for guidance
    gray = rgb2gray(I);

    % Apply guided filtering if available
    if exist('imguidedfilter', 'file') == 2
        tRefined = imguidedfilter(t, gray, ...
            'NeighborhoodSize', [31 31], ...
            'DegreeOfSmoothing', 1e-4);
    else
        % Fallback to Gaussian smoothing
        kernel = fspecial('gaussian', [15 15], 2);
        tRefined = imfilter(t, kernel, 'replicate');
    end
end