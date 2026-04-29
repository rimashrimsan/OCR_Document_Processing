import cv2
import numpy as np
from PIL import Image
import math
from sklearn.cluster import KMeans

# Safety wrapper for MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

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
    """Detect dominant page color using K-Means."""
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

    dominant_cluster = np.bincount(kmeans.labels_).argmax()
    return kmeans.cluster_centers_[dominant_cluster].astype(int)


def create_page_mask(image, page_color, tolerance=40):
    """Create a binary mask of pixels matching the page color."""
    diff = np.abs(image.astype(float) - page_color.astype(float))
    mask = np.all(diff <= tolerance, axis=2).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def find_largest_page_contour(mask):
    """Find corners of the largest page-like object."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < (mask.shape[0] * mask.shape[1] * 0.15):
        return None

    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    if len(approx) == 4: return approx.reshape(4, 2)

    x, y, w, h = cv2.boundingRect(largest_contour)
    return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])


def order_points(pts):
    """Order 4 points: tl, tr, br, bl."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """Warp image to flatten document."""
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    height = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    if width < 10 or height < 10: return image
    dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))


def detect_hands_ai(image):
    """Detect hands using MediaPipe AI with safety check."""
    if not MEDIAPIPE_AVAILABLE: return None
    detector = get_hands_detector()
    if detector is None: return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try: results = detector.process(image_rgb)
    except Exception: return None
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark])
            hull = cv2.convexHull(points)
            cv2.drawContours(mask, [hull], -1, 255, -1)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)
    return mask


def detect_skin_mask(image):
    """Fallback skin detection."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ranges = [(np.array([0, 20, 70]), np.array([20, 150, 255])), (np.array([0, 30, 60]), np.array([30, 170, 255]))]
    skin_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges: skin_mask = cv2.bitwise_or(skin_mask, cv2.inRange(hsv, lower, upper))
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(skin_mask)
    for c in contours:
        if cv2.contourArea(c) > 500: cv2.drawContours(filtered, [c], -1, 255, -1)
    return filtered


def remove_shadows_and_bleed(image):
    """Estimate background field and subtract shadows."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    bg = cv2.medianBlur(cv2.dilate(l, np.ones((7, 7))), 21)
    diff = 255 - cv2.absdiff(l, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(cv2.merge((norm, a, b)), cv2.COLOR_LAB2BGR)


def enhance_contrast_clahe(image):
    """CLAHE enhancement."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def auto_rotate(image):
    """Deskew text orientation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    if lines is None: return image
    angles = [np.degrees(np.arctan2(y2-y1, x2-x1)) for line in lines for x1,y1,x2,y2 in line if abs(np.degrees(np.arctan2(y2-y1, x2-x1))) < 45]
    if not angles: return image
    angle = np.median(angles)
    if abs(angle) < 0.5 or abs(angle) > 15: return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def auto_white_balance(image):
    """Gray World White Balance."""
    res = image.copy().astype(np.float32)
    avg = np.mean(res, axis=(0,1))
    avg_all = np.mean(avg)
    for i in range(3): res[:,:,i] = np.clip(res[:,:,i] * (avg_all / max(avg[i], 1)), 0, 255)
    return res.astype(np.uint8)


def remove_borders(image, border_percent=2):
    """Crop edges."""
    h, w = image.shape[:2]
    ch, cw = int(h * border_percent / 100), int(w * border_percent / 100)
    return image[ch:h-ch, cw:w-cw] if ch > 0 and cw > 0 else image


def detect_table_grid(image):
    """Find grid lines."""
    binary = cv2.adaptiveThreshold(~cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 40, 1)), iterations=2)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, image.shape[0] // 40)), iterations=2)
    return cv2.add(h_lines, v_lines)


def smart_scan_document(image_bgr, **kwargs):
    """Orchestrate scanning pipeline."""
    res = image_bgr.copy()
    pts = find_largest_page_contour(create_page_mask(res, get_dominant_page_color(res), kwargs.get('crop_tolerance', 50)))
    if pts is not None:
        res = four_point_transform(res, pts) if kwargs.get('deskew', True) and len(pts)==4 else res[cv2.boundingRect(pts)[1]:cv2.boundingRect(pts)[1]+cv2.boundingRect(pts)[3], cv2.boundingRect(pts)[0]:cv2.boundingRect(pts)[0]+cv2.boundingRect(pts)[2]]
    if kwargs.get('border_cleanup', True): res = remove_borders(res)
    if kwargs.get('white_balance_enabled', True): res = auto_white_balance(res)
    if kwargs.get('auto_rotate_enabled', True): res = auto_rotate(res)
    if kwargs.get('remove_hands', True):
        mask = detect_hands_ai(res)
        if mask is None or not np.any(mask): mask = detect_skin_mask(res)
        if mask is not None and np.any(mask): res = cv2.inpaint(res, cv2.dilate(mask, np.ones((3,3)), iterations=2), 3, cv2.INPAINT_TELEA)
    if kwargs.get('fix_shadows', True): res = remove_shadows_and_bleed(res)
    if kwargs.get('enhance_contrast', True): res = enhance_contrast_clahe(res)
    if kwargs.get('bw_mode', False): res = cv2.cvtColor(cv2.adaptiveThreshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10), cv2.COLOR_GRAY2BGR)
    
    table = detect_table_grid(res) if kwargs.get('detect_tables', False) else None
    return (res, table) if kwargs.get('detect_tables', False) else res
