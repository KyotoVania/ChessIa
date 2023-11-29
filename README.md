# Chess AI Project

## Description
The Chess AI project is a Python-based application that leverages neural network capabilities using PyTorch. It offers an engaging platform for users to play chess against an AI, observe AI vs. AI games, or compete against an AI with random moves.

## Features
- **Play Chess Against an AI**: Test your skills by playing against a computer-controlled opponent.
- **AI vs. AI**: Sit back and watch as two AI opponents battle it out on the chessboard.
- **Random Move AI**: Enjoy a less predictable game by playing against an AI that makes random moves.
- **Continuous Game Mode**: Play games continuously with the option to pause, change modes, or exit at any time.
- **Verbose Mode**: Get detailed game output for an in-depth understanding of each move.

## Requirements
- Python 3.x
- PyTorch
- python-chess library

## Installation
1. **Clone the Repository**:
   ```bash
   git clone [URL]

2. **Install Required Packages**:
   ```bash
   pip install torch chess

## Usage

Run the program from the command line, specifying the mode and additional options:

``` bash 
python chess_ai.py --mode [MODE] [--continuous] [--verbose]

Modes
ai_vs_human: Engage in a strategic game against the AI.
ai_vs_ai: Watch an automated chess match between two AI players.
ai_vs_random: Challenge an AI opponent that moves randomly.
Options
--continuous: Continuously play games until manually stopped. Options to continue, change the mode, or exit are available after stopping.
--verbose: Enable verbose mode for detailed game insights, including the chessboard state after each move.
```

## Examples
Play a Single Game Against the AI:
```bash
python chess_ai.py --mode ai_vs_human
```
Run Continuous AI vs. AI Games:
```bash
python chess_ai.py --mode ai_vs_ai --continuous
```
Play a Verbose Game Against a Random-Moving AI:
```bash
python chess_ai.py --mode ai_vs_random --verbose