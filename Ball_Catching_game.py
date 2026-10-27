
import cv2
import mediapipe
import numpy
import random

screen_width = 800
screen_height = 600
panel_width = 300
camera_height = 200

ball_radius = 20
ball_speed = 10
ball_x = 0
ball_y = 0

basket_width = 100
basket_height = 20
basket_x = screen_width // 2
basket_y = screen_height - 40
basket_left = basket_x - basket_width // 2
basket_right = basket_x + basket_width // 2

score = 0
misses = 0
max_misses = 3
started = False
game_over = False

hands = mediapipe.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def create_ball():
    global ball_x, ball_y
    ball_x = random.randint(ball_radius, screen_width - ball_radius)
    ball_y = ball_radius

def move_ball():
    global ball_y
    ball_y += ball_speed

def move_basket(frame):
    global basket_x, basket_left, basket_right

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mediapipe.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mediapipe.solutions.hands.HAND_CONNECTIONS)

        index_tip = hand_landmarks.landmark[8]
        basket_x = int(index_tip.x * screen_width)
        basket_x = max(basket_width // 2, min(basket_x, screen_width - basket_width // 2))

    basket_left = basket_x - basket_width // 2
    basket_right = basket_x + basket_width // 2

    return frame

def check_collision():
    global score, misses, ball_speed, game_over

    if ball_y + ball_radius >= basket_y and ball_y - ball_radius <= basket_y + basket_height and abs(ball_x - basket_x) <= basket_width // 2 + ball_radius:
        score += 1
        ball_speed += 1
        create_ball()
    
    elif ball_y - ball_radius > screen_height:
        misses += 1
        create_ball()
        if misses >= max_misses:
            game_over = True

def game_screen(frame):
    screen = numpy.zeros((screen_height, screen_width + panel_width, 3), dtype=numpy.uint8)
    screen[:, :] = (35, 35, 35)
    screen[:camera_height, screen_width:] = cv2.resize(frame, (panel_width, camera_height))
    screen[camera_height:, screen_width:] = (45, 45, 45)

    if not started:
        cv2.putText(screen, "CATCH THE BALL", (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.rectangle(screen, (300, 250), (500, 320), (0, 180, 0), -1)
        cv2.putText(screen, "START", (350, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(screen, "Press S to start", (280, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    else:
        cv2.circle(screen, (ball_x, ball_y), ball_radius, (0, 200, 255), -1)
        cv2.rectangle(screen, (basket_left, basket_y), (basket_right, basket_y + basket_height), (0, 255, 0), -1)
        cv2.putText(screen, "CATCH THE BALL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(screen, "YOUR SCORE", (820, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(screen, f"{score}", (820, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(screen, "MISSES", (820, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(screen, f"{misses} / {max_misses}", (820, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 3)

        if game_over:
            cv2.rectangle(screen, (100, 220), (700, 370), (20, 20, 20), -1)
            cv2.putText(screen, "GAME OVER", (220, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.putText(screen, f"Final Score: {score}", (280, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(screen, "Press Q to quit", (820, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.rectangle(screen, (screen_width, 0), (screen_width + panel_width - 1, camera_height - 1), (255, 255, 255), 2)

    return screen

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong mo duoc webcam!")
    hands.close()
    raise SystemExit

create_ball()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc hinh anh tu webcam!")
        break

    frame = move_basket(frame)

    if started and not game_over:
        move_ball()
        check_collision()

    screen = game_screen(frame)
    cv2.imshow("Ball Catching Game", screen)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('s') and not started:
        started = True
        create_ball()

    if key == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
