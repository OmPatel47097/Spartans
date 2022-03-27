import cv2 as cv
import mediapipe as mp

Mp_drawing = mp.solutions.drawing_utils
Mp_drawing_style = mp.solutions.drawing_styles
Mp_hands = mp.solutions.hands

def getHendMove(hand_landmarks):
    landmarks = hand_landmarks.landmark
    if all([landmarks[i].y < landmarks[i+3].y for i in range(9, 20, 4)]): return "Rock"
    elif landmarks[13].y < landmarks[16].y and landmarks[17].y < landmarks[20].y: return "Secissors"
    else: return "paper"

vod = cv.VideoCapture(0)

clock = 0
p_one = p_two = None
gameText = ""
success = True

with Mp_hands.Hands(model_complexity = 0,
                    min_detection_confidence = 0.5,
                    min_tracking_confidence = 0.5
                    ) as hands:
    while True:
        ret, frame = vod.read()
        if not ret or frame is None: break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                Mp_drawing.draw_landmarks(frame,
                                          hand_landmarks,
                                          Mp_hands.HAND_CONNECTIONS,
                                          Mp_drawing_style.get_default_hand_landmarks_style(),
                                          Mp_drawing_style.get_default_hand_connections_style())

        frame = cv.flip(frame, 1)

        if 0 <= clock < 20:
            success = True
            gameText = "Ready?"
        elif clock < 30: gameText = "3..."
        elif clock < 40: gameText = "2..."
        elif clock < 50: gameText = "1..."
        elif clock < 60: gameText = "Goo!"
        elif clock == 60:
            hls = results.multi_hand_landmarks
            if hls and len(hls) == 2:
                p1_move = getHendMove(hls[0])
                p2_move = getHendMove(hls[1])
            else:
                success = False
        elif clock < 100:
            if success:
                gameText = f"First Player Plaied {p1_move}. Secound Player Plaied {p2_move}"
                if p1_move == p2_move: gameText = f"{gameText} Game is tied."
                elif p1_move == "paper" and p2_move == "rock" : gameText = f"{gameText} Player 1 Winss..."
                elif p1_move== "rock" and p2_move == "scissors" : gameText = f"{gameText} Player 1 Winss..."
                elif p1_move == "scissors" and p2_move == "paper" : gameText = f"{gameText} Player 1 Winss..."
                elif p1_move == "lizard" and p2_move == "spock" : gameText = f"{gameText} Player 1 Winss..."
                elif p1_move == "spock" and p2_move == "rock" : gameText = f"{gameText} Player 1 Winss..."
                elif p1_move == "spock" and p2_move == "scissors" : gameText = f"{gameText} Player 1 Winss..."
                elif p1_move == "scissors" and p2_move == "lizard" : gameText = f"{gameText} Player 1 Winss..."
                elif p1_move == "rock" and p2_move == "lizard" : gameText = f"{gameText} Player 1 Winss..."
                elif p1_move == "paper" and p2_move == "spock" : gameText = f"{gameText} Player 1 Winss..."
                elif p1_move == "lizard" and p2_move == "paper" : gameText = f"{gameText} Player 1 Winss..."
                else: gameText = f"{gameText} Player 2 Winss..."

            else:
                gameText = "Error Code: 69 / Always wear Helmate. <3"

        cv.imshow('frame',frame)

        if cv.waitKey(1) & 0xFF == ord('q'): break

vod.release()
cv.destroyAllWindows()
