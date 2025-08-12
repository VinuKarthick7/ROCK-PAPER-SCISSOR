import random
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
import numpy as np



# ------------------------------------------------------------------
# CONFIG – fill the whole display
# ------------------------------------------------------------------
SCREEN_W, SCREEN_H = 1920, 1080        # <-- set to your monitor (e.g. 1366, 768)
WIN_SCORE = 5                           # first to 5 wins the match



# ------------------------------------------------------------------
# UI helper functions  (from the display build)
# ------------------------------------------------------------------
def draw_static_ui(bg):
    # Removed title text "ROCK  -  PAPER  -  SCISSORS"
    return


def draw_dynamic_ui(bg, scores, timer, state_result, start_game):
    # Instructions
    cv2.putText(bg, "Press  S  to play a round", (30, 95),
                cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 3)
    if start_game:
        cv2.putText(bg, "Press  R  to reset", (30, 130),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    # Countdown digits while waiting for result
    if start_game and not state_result:
        col = (255, 255, 255) if timer < 1 else (0, 255, 255) if timer < 2 else (0, 0, 255)
        cv2.putText(bg, f"{int(timer)}", (605, 435),
                    cv2.FONT_HERSHEY_PLAIN, 6, col, 4)


def draw_round_result(bg, round_result_text):
    if round_result_text:
        cv2.putText(bg, round_result_text, (30, 165),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)


def rect_from_rel(img, x, y, w, h):
    """Convert relative rect → absolute pixel bounds."""
    H, W = img.shape[:2]
    x1 = int(W * x); y1 = int(H * y)
    x2 = int(W * (x + w)); y2 = int(H * (y + h))
    return x1, y1, x2, y2


def paste_panel(bg, rel, content):
    """Resize content to rel-panel and paste/overlay to bg."""
    x1, y1, x2, y2 = rect_from_rel(bg, *rel)
    tw, th = x2 - x1, y2 - y1
    if content is None or tw <= 0 or th <= 0:
        return bg
    resized = cv2.resize(content, (tw, th), interpolation=cv2.INTER_AREA)
    if resized.shape[-1] == 4:  # PNG with alpha
        return cvzone.overlayPNG(bg, resized, (x1, y1))
    bg[y1:y2, x1:x2] = resized
    return bg



# ------------------------------------------------------------------
# BACK-END initialisation  (original logic + tweaks)
# ------------------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 620); cap.set(4, 480)

cv2.namedWindow("BG", cv2.WINDOW_NORMAL)                  # fullscreen, no borders
cv2.setWindowProperty("BG", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.moveWindow("BG", 0, 0)

detector = HandDetector(maxHands=1)

timer, stateResult, startGame = 0, False, False
round_result_text = ""                                     # per-round message
scores = [0, 0]                                            # [AI, Player]
imgAI = None

# 1=Rock, 2=Paper, 3=Scissors
beat = {1: 2, 2: 3, 3: 1}                                  # mapping: what beats the key

# Markov 3×3×3 (prev two player moves ➜ next player move distribution)
transition_matrix = np.ones((3, 3, 3), dtype=np.float64)
prev_moves_list = []                                       # last two player moves

# --- asset loading -------------------------------------------------
imgBG_original = cv2.imread("images/BG.png")
ai_images = {
    1: cv2.imread("images/1.png", cv2.IMREAD_UNCHANGED),
    2: cv2.imread("images/2.png", cv2.IMREAD_UNCHANGED),
    3: cv2.imread("images/3.png", cv2.IMREAD_UNCHANGED),
}
imgVictory  = cv2.imread("images/player_wins_screen.png")
imgGameOver = cv2.imread("images/ai_wins_screen.png")
for tag, ref in {"BG": imgBG_original,
                 "Victory": imgVictory,
                 "GameOver": imgGameOver}.items():
    if ref is None:
        raise FileNotFoundError(f"Missing image for {tag}")



# ------------------------------------------------------------------
# BACK-END functions  (unchanged core + small guards)
# ------------------------------------------------------------------
def normalize_transition_matrix():
    for i in range(3):
        for j in range(3):
            s = transition_matrix[i, j].sum()
            if s:
                transition_matrix[i, j] /= s


def update_transition(prev, cur):
    if len(prev) >= 2:
        prev1, prev2 = prev
        transition_matrix[prev1-1, prev2-1, cur-1] += 1
    normalize_transition_matrix()


def ai_predict():
    # Either not enough history or exploration
    if len(prev_moves_list) < 2 or random.random() < 0.3:
        return random.randint(1, 3)
    p1, p2 = prev_moves_list[-2:]
    predicted_player = int(np.argmax(transition_matrix[p1-1, p2-1])) + 1
    return beat[predicted_player]


def recognize_player_move(hands):
    """Return 1=Rock, 2=Paper, 3=Scissors, or None if no hand/unknown pattern."""
    if not hands:
        return None
    finger_pattern = detector.fingersUp(hands[0])  # [thumb, index, middle, ring, pinky], 1=open, 0=closed
    if finger_pattern == [0, 0, 0, 0, 0]:
        return 1  # Rock
    elif finger_pattern == [1, 1, 1, 1, 1]:
        return 2  # Paper
    elif finger_pattern == [0, 1, 1, 0, 0]:
        return 3  # Scissors
    return None



# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------
def main():
    global timer, stateResult, startGame, scores, imgAI
    global prev_moves_list, transition_matrix, round_result_text, initialTime

    ai_rel  = (0.12, 0.34, 0.31, 0.54)     # AI hand panel (left)
    cam_rel = (0.63, 0.34, 0.31, 0.54)     # Player cam panel (right)

    while True:
        bg = imgBG_original.copy()

        ok, frame = cap.read()
        if not ok:
            cv2.putText(bg, "Camera not available", (40, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow("BG", cv2.resize(bg, (SCREEN_W, SCREEN_H)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # --- preprocess camera frame (same crop as original code) ----
        cam = cv2.resize(frame, (0, 0), fx=0.875, fy=0.875)
        cam = cam[:, 80:480]

        hands, _ = detector.findHands(cam)  # detect (no draw)

        # --- UI layers ------------------------------------------------
        draw_static_ui(bg)

        # If a match is ongoing (someone hasn't reached WIN_SCORE)
        match_over = (scores[0] >= WIN_SCORE or scores[1] >= WIN_SCORE)

        if startGame and not match_over:
            # Update countdown
            if not stateResult:
                timer = time.time() - initialTime
            draw_dynamic_ui(bg, scores, timer, stateResult, startGame)

            # evaluate after 3 s
            if not stateResult and timer > 3:
                stateResult, timer = True, 0

                # --- recognize player move ---------------------------
                playerMove = recognize_player_move(hands)

                # --- update Markov, AI move, scores ------------------
                if playerMove:
                    update_transition(prev_moves_list, playerMove)
                    prev_moves_list.append(playerMove)
                    if len(prev_moves_list) > 2:
                        prev_moves_list.pop(0)

                aiMove = ai_predict()
                imgAI = ai_images.get(aiMove, None)

                # decide winner and set round_result_text
                if not playerMove:
                    round_result_text = "No hand detected - Round void"
                else:
                    if (playerMove == 1 and aiMove == 3) or \
                       (playerMove == 2 and aiMove == 1) or \
                       (playerMove == 3 and aiMove == 2):
                        scores[1] += 1
                        round_result_text = "Human wins the round"
                    elif (aiMove == 1 and playerMove == 3) or \
                         (aiMove == 2 and playerMove == 1) or \
                         (aiMove == 3 and playerMove == 2):
                        scores[0] += 1
                        round_result_text = "AI wins the round"
                    else:
                        round_result_text = "Draw"

        else:
            # Not counting down; still draw the UI scaffold
            draw_dynamic_ui(bg, scores, timer, stateResult, startGame)

        # Show the last round's message between rounds
        draw_round_result(bg, round_result_text)

        # --- panel rendering -----------------------------------------
        bg = paste_panel(bg, cam_rel, cam)          # player camera (right)
        bg = paste_panel(bg, ai_rel,  imgAI)        # AI hand (left)

        # scores
        cv2.putText(bg, str(scores[0]), (410, 215),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
        cv2.putText(bg, str(scores[1]), (1112, 215),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

        # match end screen at 5
        if scores[0] >= WIN_SCORE:
            bg = imgGameOver.copy()
            startGame = False
        elif scores[1] >= WIN_SCORE:
            bg = imgVictory.copy()
            startGame = False

        # full-screen display
        cv2.imshow("BG", cv2.resize(bg, (SCREEN_W, SCREEN_H)))

        # keyboard control
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            # Only start a new round if match not over
            if scores[0] < WIN_SCORE and scores[1] < WIN_SCORE:
                startGame = True
                stateResult = False
                initialTime = time.time()
                timer = 0
                # Optional: clear last message when countdown starts
                # round_result_text = ""
        elif k == ord("r"):
            scores = [0, 0]
            prev_moves_list.clear()
            transition_matrix[:] = 1
            normalize_transition_matrix()
            imgAI, stateResult, timer = None, False, 0
            startGame = False
            round_result_text = ""
        elif k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



# -----------------------------------------------------------------
# ENTRY-POINT
# -----------------------------------------------------------------
if __name__ == "__main__":
    main()
