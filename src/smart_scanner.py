import cv2
import numpy as np
from sklearn.cluster import KMeans

# Global variables for AI detection (initialized lazily)
_MEDIAPIPE_HANDS = None
_HANDS_DETECTOR = None
_MEDIAPIPE_FAILED = False

def get_hands_detector():
    """Lazily initialize MediaPipe to avoid startup crashes."""
    global _MEDIAPIPE_HANDS, _HANDS_DETECTOR, _MEDIAPIPE_FAILED
    
    if _MEDIAPIPE_FAILED:
        return None
    
    if _HANDS_DETECTOR is not None:
        return _HANDS_DETECTOR
        
    try:
        import mediapipe as mp
        _MEDIAPIPE_HANDS = mp.solutions.hands
        _HANDS_DETECTOR = _MEDIAPIPE_HANDS.Hands(
            static_image_mode=True,
            max_num_hands=4,
            min_detection_confidence=0.5
        )
        return _HANDS_DETECTOR
    except Exception:
        _MEDIAPIPE_FAILED = True
        return None


def get_dominant_page_color(image):
    """
    Detect the dominant page color by sampling from multiple regions.
    Uses K Means clustering to find the most common color.
    """
    h, w = image.shape[:2]
    samples = []
    margin = int(min(h, w) * 0.1)

    center_region = image[h//4:3*h//4, w//4:3*w//4]
    samples.append(center_region)

    if h > margin * 2 + 100:
        top_region = image[margin:margin+100, w//3:2*w//3]
        samples.append(top_region)

    if h > margin * 2 + 100:
        bottom_region = image[h-margin-100:h-margin, w//3:2*w//3]
        samples.append(bottom_region)

    combined = np.vstack([s.reshape(-1, 3) for s in samples if s.size > 0])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(combined)

    labels = kmeans.labels_
    counts = np.bincount(labels)
    dominant_cluster = counts.argmax()
    dominant_color = kmeans.cluster_centers_[dominant_cluster]

    return dominant_color.astype(int)


def create_page_mask(image, page_color, tolerance=40):
    """Create a binary mask of pixels matching the page color."""
    diff = np.abs(image.astype(float) - page_color.astype(float))
    mask = np.all(diff <= tolerance, axis=2).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def find_largest_page_contour(mask):
    """
    Find the largest contour and approximate to a polygon.
    Returns 4 corner points for perspective correction.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    image_area = mask.shape[0] * mask.shape[1]
    if cv2.contourArea(largest_contour) < image_area * 0.15:
        return None

    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2)

    x, y, w, h = cv2.boundingRect(largest_contour)
    return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])


def order_points(pts):
    """Order 4 points as: top left, top right, bottom right, bottom left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """Apply perspective warp to flatten a tilted document."""
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
    widthB = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))
    heightB = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth < 10 or maxHeight < 10:
        return image

    dst = np.array([
        [0, 0], [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1], [0, maxHeight-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def detect_hands_ai(image):
    """
    Detect hands using MediaPipe AI landmarks.
    Works for all skin tones and is much more accurate than color filtering.
    """
    detector = get_hands_detector()
    if detector is None:
        return None

    # MediaPipe uses RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        results = detector.process(image_rgb)
    except Exception:
        return None
    
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get points for the hand
            points = []
            for lm in hand_landmarks.landmark:
                points.append([int(lm.x * w), int(lm.y * h)])
            
            # Create a hull or polygon around the points
            points = np.array(points)
            hull = cv2.convexHull(points)
            cv2.drawContours(mask, [hull], -1, 255, -1)
            
            # Dilate slightly to cover the whole hand/fingers
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.dilate(mask, kernel, iterations=2)
            
    return mask


def detect_skin_mask(image):
    """Detect skin tones using multiple HSV ranges (Fallback)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ranges = [
        (np.array([0, 20, 70]), np.array([20, 150, 255])),
        (np.array([0, 30, 60]), np.array([30, 170, 255])),
        (np.array([0, 15, 40]), np.array([25, 100, 200])),
    ]
    skin_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        skin_mask = cv2.bitwise_or(skin_mask, cv2.inRange(hsv, lower, upper))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(skin_mask)
    for c in contours:
        if cv2.contourArea(c) > 500:
            cv2.drawContours(filtered, [c], -1, 255, -1)
    return filtered


def inpaint_region(image, mask):
    """Fill masked regions using Telea Fast Marching inpainting."""
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    return cv2.inpaint(image, dilated, 3, cv2.INPAINT_TELEA)


def remove_shadows_and_bleed(image):
    """
    Remove uneven lighting, phone shadows, and back page bleed through
    by estimating the background illumination field.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    dilated = cv2.dilate(l, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff_img = 255 - cv2.absdiff(l, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result = cv2.merge((norm_img, a, b))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def enhance_contrast_clahe(image, clip_limit=2.0):
    """Apply CLAHE to make text sharper and more readable."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    result = cv2.merge((cl, a, b))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def auto_rotate(image):
    """
    Detect dominant text line angle using Hough Transform
    and rotate the image to make text perfectly horizontal.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5 or abs(median_angle) > 15:
        return image

    h, w = image.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def denoise(image, strength=10):
    """Remove speckle noise using Non Local Means Denoising."""
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def sharpen(image):
    """Apply unsharp mask to make text edges crisp."""
    gaussian = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)


def binarize_document(image):
    """Convert to clean black and white using adaptive thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def auto_white_balance(image):
    """
    Apply automatic white balance correction using the Gray World algorithm.
    Corrects color casts from artificial lighting (yellow lamps, blue screens).
    """
    result = image.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_all / max(avg_b, 1)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_all / max(avg_g, 1)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_all / max(avg_r, 1)), 0, 255)

    return result.astype(np.uint8)


def remove_borders(image, border_percent=2):
    """
    Remove dark borders and edges that appear after perspective correction
    or from scanner glass edges.
    """
    h, w = image.shape[:2]
    crop_h = int(h * border_percent / 100)
    crop_w = int(w * border_percent / 100)
    if crop_h > 0 and crop_w > 0:
        return image[crop_h:h-crop_h, crop_w:w-crop_w]
    return image


def detect_table_grid(image):
    """
    Detect tables by looking for horizontal and vertical lines.
    Returns a binary mask of the table grid.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    
    # Isolate horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 40, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
    
    # Isolate vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image.shape[0] // 40))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)
    
    # Combine and detect intersections
    grid = cv2.add(h_lines, v_lines)
    return grid


def smart_scan_document(
    image_bgr,
    crop_tolerance=50,
    remove_hands=True,
    enhance_contrast=True,
    deskew=True,
    fix_shadows=True,
    auto_rotate_enabled=True,
    denoise_enabled=False,
    sharpen_enabled=False,
    bw_mode=False,
    white_balance_enabled=True,
    border_cleanup=True,
    detect_tables=False
):
    """Full document scanning pipeline."""
    # Step 1: Detect and crop to page
    page_color = get_dominant_page_color(image_bgr)
    page_mask = create_page_mask(image_bgr, page_color, tolerance=crop_tolerance)
    pts = find_largest_page_contour(page_mask)

    if pts is not None:
        if deskew and len(pts) == 4:
            result = four_point_transform(image_bgr, pts)
        else:
            x, y, w, h = cv2.boundingRect(pts)
            result = image_bgr[y:y+h, x:x+w]
    else:
        result = image_bgr.copy()

    # Step 2: Clean up borders
    if border_cleanup:
        result = remove_borders(result)

    # Step 3: White balance
    if white_balance_enabled:
        result = auto_white_balance(result)

    # Step 4: Auto rotate
    if auto_rotate_enabled:
        result = auto_rotate(result)

    # Step 5: Remove hands
    if remove_hands:
        hand_mask = detect_hands_ai(result)
        if hand_mask is None or not np.any(hand_mask):
            hand_mask = detect_skin_mask(result)
        if hand_mask is not None and np.any(hand_mask):
            result = inpaint_region(result, hand_mask)

    # Step 6: Remove shadows
    if fix_shadows:
        result = remove_shadows_and_bleed(result)

    # Step 7: Denoise
    if denoise_enabled:
        result = denoise(result)

    # Step 8: Enhance contrast
    if enhance_contrast:
        result = enhance_contrast_clahe(result)

    # Step 9: Sharpen
    if sharpen_enabled:
        result = sharpen(result)

    # Step 10: Black and white mode
    if bw_mode:
        result = binarize_document(result)

    # Step 11: Table detection (Optional)
    table_grid = None
    if detect_tables:
        table_grid = detect_table_grid(result)

    if detect_tables:
        return result, table_grid
    return result