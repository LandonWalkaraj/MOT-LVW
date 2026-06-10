import argparse
import cv2


def create_csrt_tracker():
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()

    if hasattr(cv2, 'TrackerCSRT') and hasattr(cv2.TrackerCSRT, 'create'):
        return cv2.TrackerCSRT.create()

    raise RuntimeError(
        'CSRT tracker is unavailable. Install opencv-contrib-python in the selected Python environment.'
    )


def main():
    parser = argparse.ArgumentParser(description='Simple bounding box tracker for video input')
    parser.add_argument('--video', default=0, help='Path to video file or camera index (default=0)')
    args = parser.parse_args()

    try:
        source = int(args.video)
    except ValueError:
        source = args.video

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print('Unable to open video source:', args.video)
        return

    ret, frame = cap.read()
    if not ret:
        print('Unable to read frame from source')
        return

    print('Select the object to track and press ENTER or SPACE')
    bbox = cv2.selectROI('Select Object', frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow('Select Object')

    if bbox == (0, 0, 0, 0):
        print('No bounding box selected. Exiting.')
        return

    tracker = create_csrt_tracker()
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ok, bbox = tracker.update(frame)
        if ok:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Tracking', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Lost', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Bounding Box Tracker', frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
