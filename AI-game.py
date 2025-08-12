import random
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
import numpy as np

# Initialize video capture and window
cap = cv2.VideoCapture(0)
cap.set(3, 620)
cap.set(4, 480)
cv2.namedWindow("BG", cv2.WINDOW_NORMAL)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Game state variables
timer = 0
stateResult = False
startGame = False
scores = [0, 0]  # [AI, Player]
imgAI = None # Initialize imgAI outside the loop to prevent errors

# RPS Mapping: Rock = 1, Paper = 2, Scissors = 3
beat = {1: 2, 2: 3, 3: 1}

# Now using a 3x3x3 transition matrix to store transitions for a sequence of TWO moves.
# The dimensions are (Previous Move 1, Previous Move 2, Current Move).
transition_matrix = np.ones((3, 3, 3))

# Previous player move
prev_moves_list = []  # Use a list to store the last two moves

# === BEGIN: Backend Enhancement - Asset Pre-loading ===
# Load all static assets ONCE at the beginning of the program for performance.
imgBG_original = cv2.imread("images/BG.png")
ai_images = {
    1: cv2.imread('images/1.png', cv2.IMREAD_UNCHANGED),
    2: cv2.imread('images/2.png', cv2.IMREAD_UNCHANGED),
    3: cv2.imread('images/3.png', cv2.IMREAD_UNCHANGED)
}
# Load victory and game-over screens
imgVictory = cv2.imread('images/player_wins_screen.png')
imgGameOver = cv2.imread('images/ai_wins_screen.png')

# Add a check to ensure images were loaded successfully
if imgVictory is None:
    print("Error: 'player_wins_screen.png' not found or could not be loaded. Please check the file path.")
    exit()
if imgGameOver is None:
    print("Error: 'ai_wins_screen.png' not found or could not be loaded. Please check the file path.")
    exit()
# === END: Backend Enhancement - Asset Pre-loading ===

# Function to update Markov transition matrix
def update_transition(prev_moves, current):
    """Updates the Markov transition matrix based on the player's last two moves."""
    if len(prev_moves) >= 2:
        prev1, prev2 = prev_moves
        transition_matrix[prev1 - 1, prev2 - 1, current - 1] += 1
    # Normalize the matrix to get probabilities
    for i in range(3):
        for j in range(3):
            total_sum = np.sum(transition_matrix[i, j, :])
            if total_sum > 0:
                transition_matrix[i, j, :] /= total_sum

# AI Prediction using Markov Model with randomness
def ai_predict():
    """Predicts the AI's next move based on a deeper Markov model."""
    # Add a 30% chance for a random move to keep the AI unpredictable
    if len(prev_moves_list) < 2 or random.random() < 0.3:
        return random.randint(1, 3)

    # Use the last two moves to predict the next one
    prev1, prev2 = prev_moves_list[-2], prev_moves_list[-1]
    
    # Predict next move based on the last two moves
    predicted_next_move = np.argmax(transition_matrix[prev1 - 1, prev2 - 1]) + 1
    
    # The AI plays the counter move to the predicted player move
    return beat[predicted_next_move]

# === NEW: Main game loop refactored into a function ===
def main():
    global timer, stateResult, startGame, scores, imgAI, prev_moves_list, transition_matrix, imgBG_original, cap, imgGameOver, imgVictory, initialTime

    while True:
        imgBG = imgBG_original.copy()
        success, img = cap.read()

        imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
        imgScaled = imgScaled[:, 80:480]

        hands, img = detector.findHands(imgScaled)  # Detect hands

        if startGame:
            cv2.putText(imgBG, 'Press r to reset', (490, 115),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 4)
            if not stateResult:
                timer = time.time() - initialTime
                
                # === NEW: Dynamic timer color based on the time ===
                if timer < 1:
                    timer_color = (255, 255, 255) # White
                elif timer < 2:
                    timer_color = (0, 255, 255) # Yellow
                else:
                    timer_color = (0, 0, 255) # Red
                
                cv2.putText(imgBG, str(int(timer)), (605, 435),
                            cv2.FONT_HERSHEY_PLAIN, 6, timer_color, 4)
                # ===================================================

                if timer > 3:
                    stateResult = True
                    timer = 0
                    
                    if hands:
                        playerMove = None
                        hand = hands[0]
                        fingers = detector.fingersUp(hand)
                        if fingers == [0, 0, 0, 0, 0]:
                            playerMove = 1
                        elif fingers == [1, 1, 1, 1, 1]:
                            playerMove = 2
                        elif fingers == [0, 1, 1, 0, 0]:
                            playerMove = 3

                        print(f"Player Move: {playerMove}")

                        if playerMove:
                            update_transition(prev_moves_list, playerMove)
                            prev_moves_list.append(playerMove)
                            
                            if len(prev_moves_list) > 2:
                                prev_moves_list.pop(0)

                        aiMove = ai_predict()

                        imgAI = ai_images[aiMove]
                        imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

                        # Determine winner
                        if playerMove:
                            if (playerMove == 1 and aiMove == 3) or \
                                    (playerMove == 2 and aiMove == 1) or \
                                    (playerMove == 3 and aiMove == 2):
                                scores[1] += 1
                            elif (playerMove == 3 and aiMove == 1) or \
                                    (playerMove == 1 and aiMove == 2) or \
                                    (playerMove == 2 and aiMove == 3):
                                scores[0] += 1

        imgBG[234:654, 795:1195] = imgScaled

        if stateResult and imgAI is not None:
            imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

        cv2.putText(imgBG, str(scores[0]), (410, 215),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
        cv2.putText(imgBG, str(scores[1]), (1112, 215),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

        # Check for winner
        if scores[0] == 10:
            imgBG = imgGameOver.copy()
            startGame = False
        elif scores[1] == 10:
            imgBG = imgVictory.copy()
            startGame = False

        cv2.imshow("BG", imgBG)

        key = cv2.waitKey(1)
        if key == ord('s'):
            startGame = True
            initialTime = time.time()
            stateResult = False
        elif key == ord('r'):
            scores = [0, 0]  # Reset
            prev_moves_list = []
            transition_matrix = np.ones((3, 3, 3))
            for i in range(3):
                for j in range(3):
                    transition_matrix[i, j, :] /= np.sum(transition_matrix[i, j, :])
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
