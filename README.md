# Rock-Paper-Scissors AI Game 🎮✋🤖

An intelligent Rock-Paper-Scissors game that uses computer vision for real-time hand gesture recognition and an adaptive AI opponent powered by Markov Chain machine learning algorithms.

## 🌟 Project Highlights

- **Real-time Computer Vision**: Recognizes hand gestures through your webcam using OpenCV and MediaPipe
- **Adaptive AI Opponent**: Learns from your playing patterns using a 3D Markov Chain model
- **Interactive Gameplay**: Fullscreen gaming experience with visual feedback and score tracking
- **Machine Learning**: AI predicts and counters player moves based on historical patterns

## 🚀 Features

### Computer Vision & Gesture Recognition
- Live hand tracking and detection
- Recognizes three gestures:
  - 🪨 **Rock**: Closed fist
  - 📄 **Paper**: Open hand with all fingers extended
  - ✂️ **Scissors**: Index and middle fingers extended
- Real-time processing at 30+ FPS

### Intelligent AI
- **Pattern Learning**: Tracks your last two moves to predict the next one
- **Markov Chain Model**: Uses a 3×3×3 transition matrix to model player behavior
- **Adaptive Strategy**: Dynamically updates predictions after each round
- **Exploration-Exploitation**: 30% random moves to avoid being too predictable

### Game Features
- Best of 5 rounds (first to reach 5 points wins)
- Real-time countdown timer (3 seconds per round)
- Score tracking for both player and AI
- Visual feedback for round results
- Custom victory/defeat screens
- Fullscreen immersive experience

## 🛠️ Technologies Used

- **Python 3.x** - Core programming language
- **OpenCV** - Computer vision and image processing
- **cvzone** - Hand tracking module
- **MediaPipe** - Hand landmark detection
- **NumPy** - Numerical computations for AI model

## 📋 Prerequisites

- Python 3.7 or higher
- Webcam connected to your computer
- Operating System: Windows, macOS, or Linux

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/VinuKarthick7/ROCK-PAPER-SCISSOR.git
   cd ROCK-PAPER-SCISSOR
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python --version  # Should be 3.7+
   pip list  # Verify all packages are installed
   ```

## 🎮 How to Play

1. **Start the game**
   ```bash
   python AI-game.py
   ```
   or
   ```bash
   python test.py
   ```

2. **Game Controls**
   - Press **'S'** to start a new round
   - Press **'R'** to reset the game and scores
   - Press **'Q'** to quit the game

3. **Playing a Round**
   - Position your hand in front of the webcam
   - When countdown starts, make your gesture (rock, paper, or scissors)
   - Hold your gesture until the countdown completes
   - See the result and updated scores

4. **Winning the Game**
   - First player to reach 5 points wins the match
   - Victory or defeat screen will be displayed
   - Press 'R' to start a new match

## 🎯 How the AI Works

### Markov Chain Model
The AI uses a probabilistic model that:
1. **Observes** your last two moves (e.g., Rock → Paper)
2. **Predicts** your likely next move based on historical patterns
3. **Counters** by choosing the gesture that beats your predicted move
4. **Learns** by updating probabilities after each round

### Example
If you frequently play: Rock → Paper → Scissors
- The AI learns this pattern in its transition matrix
- It predicts you'll play Scissors after Rock → Paper
- It counters by choosing Rock (which beats Scissors)

### Exploration Strategy
- 30% of the time, AI makes random moves to avoid being predictable
- 70% of the time, AI uses learned patterns to counter your strategy

## 📁 Project Structure

```
ROCK-PAPER-SCISSOR/
├── AI-game.py              # Main game with enhanced UI
├── test.py                 # Alternative version with title display
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── PROJECT_SUMMARY.md     # Detailed technical documentation
└── images/                # Game assets
    ├── BG.png            # Background image
    ├── 1.png             # Rock gesture (AI)
    ├── 2.png             # Paper gesture (AI)
    ├── 3.png             # Scissors gesture (AI)
    ├── player_wins_screen.png
    └── ai_wins_screen.png
```

## 🧠 Key Algorithms

### Gesture Recognition
```python
def recognize_player_move(hands):
    finger_pattern = detector.fingersUp(hands[0])
    if finger_pattern == [0, 0, 0, 0, 0]:  # All closed
        return 1  # Rock
    elif finger_pattern == [1, 1, 1, 1, 1]:  # All open
        return 2  # Paper
    elif finger_pattern == [0, 1, 1, 0, 0]:  # Index & middle
        return 3  # Scissors
```

### AI Prediction
```python
def ai_predict():
    if len(prev_moves_list) < 2 or random.random() < 0.3:
        return random.randint(1, 3)  # Random exploration
    
    # Use Markov model to predict
    p1, p2 = prev_moves_list[-2:]
    predicted_player = argmax(transition_matrix[p1-1, p2-1])
    return beat[predicted_player]  # Counter strategy
```

## 📊 Technical Details

- **AI Model**: 3D Markov Chain (3×3×3 transition matrix)
- **Learning Method**: Online learning with incremental updates
- **Prediction Strategy**: Maximum likelihood estimation
- **Performance**: Real-time processing at 30+ FPS
- **Code Lines**: ~280 lines of Python code

## 🐛 Troubleshooting

### Camera not detected
- Ensure your webcam is properly connected
- Check if other applications are using the camera
- Try changing the camera index in code: `cap = cv2.VideoCapture(1)`

### Gesture not recognized
- Ensure good lighting conditions
- Keep hand clearly visible in the camera frame
- Make clear, distinct gestures
- Adjust hand position if needed

### Performance issues
- Close other applications using the webcam
- Reduce screen resolution if needed (modify `SCREEN_W` and `SCREEN_H`)
- Update graphics drivers

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Vinu Karthick**
- GitHub: [@VinuKarthick7](https://github.com/VinuKarthick7)

## 🙏 Acknowledgments

- OpenCV and cvzone communities for excellent computer vision tools
- MediaPipe team for hand tracking technology
- Rock-Paper-Scissors game for the timeless entertainment

## 📚 Learn More

For detailed technical documentation, see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

**Ready to test your luck against an AI that learns from you?** 🎮

*Note: This project demonstrates practical applications of computer vision, machine learning, and game development in Python.*