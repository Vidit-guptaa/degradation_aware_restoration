function J = dehaze_single_image(inputPath, outputPath)

    I = im2double(imread(inputPath));
    if size(I, 3) == 1
        I = repmat(I, [1 1 3]);
    end

    % PARAMETERS (tuned)
    omega = 0.9;        % reduced to avoid over-dehazing
    t0 = 0.2;           % increased to reduce noise
    patchSize = 15;

    % DARK CHANNEL
    dark = getDarkChannel(I, patchSize);

    % ATMOSPHERIC LIGHT
    A = estimateAtmosphere(I, dark);

    % 🔧 FIX 1: Neutralize atmospheric light (reduces color cast)
    A = mean(A) * [1 1 1];

    % NORMALIZATION
    normI = zeros(size(I));
    for c = 1:3
        normI(:, :, c) = I(:, :, c) / max(A(c), 1e-6);
    end

    % TRANSMISSION
    transmission = 1 - omega * getDarkChannel(normI, patchSize);

    % REFINEMENT
    transmission = refineTransmission(I, transmission);

    % 🔧 FIX 2: Stronger lower bound to avoid noise
    transmission = max(transmission, t0);

    % RECOVER IMAGE
    J = zeros(size(I));
    for c = 1:3
        J(:, :, c) = (I(:, :, c) - A(c)) ./ transmission + A(c);
    end

    % CLIP VALUES
    J = min(max(J, 0), 1);

    % 🔧 FIX 3: Brightness normalization
    J = J ./ max(J(:));

    % 🔧 FIX 4: Slight gamma correction (natural look)
    J = imadjust(J, [], [], 0.9);

    % 🔧 FIX 5: Sharpen image
    J = imsharpen(J, 'Radius', 1, 'Amount', 0.8);

    % SAVE OUTPUT
    if nargin > 1 && ~isempty(outputPath)
        imwrite(J, outputPath);
    end

    % DISPLAY
    figure('Name', 'Improved Dehazing Result');
    subplot(1,2,1); imshow(I); title('Input');
    subplot(1,2,2); imshow(J); title('Improved Output');
end


function dark = getDarkChannel(I, patchSize)
    minChannel = min(I, [], 3);
    se = strel('square', patchSize);
    dark = imerode(minChannel, se);
end


function A = estimateAtmosphere(I, dark)
    [h, w, ~] = size(I);
    numPixels = h * w;
    numBright = max(round(numPixels * 0.001), 1);

    darkVec = dark(:);
    imageVec = reshape(I, numPixels, 3);

    [~, indices] = sort(darkVec, 'descend');
    topIdx = indices(1:numBright);
    brightest = imageVec(topIdx, :);

    [~, maxIdx] = max(sum(brightest, 2));
    A = brightest(maxIdx, :);
end


function tRefined = refineTransmission(I, t)
    gray = rgb2gray(I);

    if exist('imguidedfilter', 'file') == 2
        tRefined = imguidedfilter(t, gray, ...
            'NeighborhoodSize', [31 31], ...
            'DegreeOfSmoothing', 1e-4); % 🔧 stronger edge preservation
    else
        kernel = fspecial('gaussian', [15 15], 2);
        tRefined = imfilter(t, kernel, 'replicate');
    end
end