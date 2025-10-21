# Lichess Puzzle Tester for GPT Models

This project tests different GPT mini models on Lichess chess puzzles to evaluate their chess-playing capabilities.

## Models Tested

The script tests these OpenAI models in 6 different configurations (you can modify these in [puzzle_tester.py](puzzle_tester.py)):

### Base Models (No Context)
1. **gpt-4o-mini** - No hints provided
2. **gpt-3.5-turbo** - No hints provided

### With Stockfish Analysis
3. **gpt-4o-mini-context** - Given top 5 moves with evaluations from Stockfish
4. **gpt-3.5-turbo-context** - Given top 5 moves with evaluations from Stockfish

### With Legal Moves List
5. **gpt-4o-mini-legal** - Given all legal moves (alphabetically sorted, UCI format)
6. **gpt-3.5-turbo-legal** - Given all legal moves (alphabetically sorted, UCI format)

**Note:** The original request mentioned o3-mini and o4-mini, but these are not currently available OpenAI models. Update the model names in the code based on what's available in your OpenAI account.

## Features

- Loads 20 random puzzles from Lichess puzzle database CSV
- Uses Stockfish to analyze positions and provide top 5 moves with evaluations
- Tests each model in 3 variants: no context, with Stockfish analysis, with legal moves
- Validates LLM outputs to ensure they are legal moves
- Calculates accuracy metrics for each model
- Generates detailed JSON report with all results
- Interactive Jupyter notebook for reviewing results

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Lichess Puzzle Database

Download the Lichess puzzle database CSV from [https://database.lichess.org/](https://database.lichess.org/)

Look for `lichess_db_puzzle.csv.bz2` (it's a large file, ~500MB compressed, ~2GB uncompressed)

Extract the `.bz2` file to get `lichess_db_puzzle.csv`

**CSV Format:**
- Each row is a puzzle
- Columns: PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl
- FEN is the position BEFORE the opponent's move
- Moves are in UCI format (e.g., "e2e4 e7e5")
- First move is the opponent's move, remaining moves are the solution

### 3. Download Stockfish

Download Stockfish from [https://stockfishchess.org/download/](https://stockfishchess.org/download/)

Extract it and note the path to the executable.

### 4. Configure Environment Variables

Create a `.env` file in the project root and add:

```
OPENAI_API_KEY=your_openai_api_key_here
STOCKFISH_PATH=path/to/stockfish/executable
LICHESS_PUZZLE_CSV=path/to/lichess_db_puzzle.csv
```

**Example on Windows:**
```
OPENAI_API_KEY=sk-...
STOCKFISH_PATH=C:\stockfish\stockfish-windows-x86-64-avx2.exe
LICHESS_PUZZLE_CSV=C:\data\lichess_db_puzzle.csv
```

**Example on Mac/Linux:**
```
OPENAI_API_KEY=sk-...
STOCKFISH_PATH=/usr/local/bin/stockfish
LICHESS_PUZZLE_CSV=/home/user/data/lichess_db_puzzle.csv
```

### 5. Get OpenAI API Access

**IMPORTANT:** Running this script will make API calls to OpenAI which will cost money.

- Ensure you have an OpenAI API key with available credits
- 20 puzzles × 6 model configurations = 120 API calls
- Estimated cost: ~$0.03-$0.15 depending on the models used (gpt-4o-mini is cheaper than gpt-4)
- Check your usage at [https://platform.openai.com/usage](https://platform.openai.com/usage)

### 6. Run the Tests

```bash
python puzzle_tester.py
```

The script will:
- Load 50 random puzzles from the CSV file (or use sample puzzles as fallback)
- Extract FEN positions by applying the opponent's first move
- Analyze each position with Stockfish to get top 5 moves
- Test each model on all puzzles
- Generate a detailed accuracy report

## How It Works

### Puzzle Loading from CSV

The script reads the Lichess puzzle database CSV and:

1. **Randomly samples 20 puzzles** from the entire database
2. **Parses each puzzle:**
   - FEN is the position BEFORE opponent's move
   - First move in the Moves column is the opponent's move
   - Applies the opponent's move to get the actual puzzle position
   - Remaining moves are the solution
3. **Formats for testing:**
   - Presents the position AFTER opponent's move
   - Tests if LLM finds the first move of the solution

### Testing Process

For each puzzle:

1. **Extract the position** (FEN notation)
2. **Analyze with Stockfish** to get top 5 moves and evaluations
3. **Get all legal moves** sorted alphabetically
4. **Create prompts** for 3 variants:
   - Base models: Just FEN + instruction
   - Context models: FEN + top 5 moves with evals + instruction
   - Legal moves models: FEN + all legal moves (UCI format) + instruction
5. **Query each LLM** with the appropriate prompt
6. **Validate responses** to ensure they are legal moves
7. **Check correctness** against the puzzle solution

### Prompt Format

**Without context (base models):**
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Find the best move for White.
Only print the best move in the position.
```

**With Stockfish analysis (context models):**
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

Top 5 candidate moves with evaluations:
1. e2e4 (eval: 0.25)
2. d2d4 (eval: 0.20)
3. g1f3 (eval: 0.15)
4. c2c4 (eval: 0.18)
5. b1c3 (eval: 0.10)

Find the best move for White.
Only print the best move in the position.
```

**With legal moves (legal moves models):**
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

Legal moves (UCI format, alphabetically):
a2a3, a2a4, b1a3, b1c3, b2b3, b2b4, c2c3, c2c4, d2d3, d2d4, e2e3, e2e4, f2f3, f2f4, g1f3, g1h3, g2g3, g2g4, h2h3, h2h4

Find the best move for White.
Only print the best move in the position.
```

## Output

The script generates:

1. **Console output** with real-time progress and final accuracy report
2. **results.json** containing:
   - Accuracy metrics for each model
   - Detailed response data for every puzzle

### Sample Output

```
Loading 20 random puzzles from Lichess CSV...
Reading puzzles from C:\data\lichess_db_puzzle.csv...
Total puzzles in CSV: 3233632
Successfully loaded 20 puzzles from CSV

Starting model testing...

Processing puzzle 1/20...
  gpt-4o-mini: e4 (Valid: True, Correct: True)
  gpt-3.5-turbo: Nf3 (Valid: True, Correct: False)
  gpt-4o-mini-context: e4 (Valid: True, Correct: True)
  gpt-3.5-turbo-context: e4 (Valid: True, Correct: True)
  gpt-4o-mini-legal: e2e4 (Valid: True, Correct: True)
  gpt-3.5-turbo-legal: e2e4 (Valid: True, Correct: True)
  ...

================================================================================
ACCURACY REPORT
================================================================================

gpt-4o-mini:
  Total Puzzles: 20
  Correct Moves: 9 (45.00%)
  Valid Moves: 19 (95.00%)

gpt-3.5-turbo:
  Total Puzzles: 20
  Correct Moves: 7 (35.00%)
  Valid Moves: 18 (90.00%)

gpt-4o-mini-context:
  Total Puzzles: 20
  Correct Moves: 14 (70.00%)
  Valid Moves: 20 (100.00%)

gpt-3.5-turbo-context:
  Total Puzzles: 20
  Correct Moves: 12 (60.00%)
  Valid Moves: 20 (100.00%)

gpt-4o-mini-legal:
  Total Puzzles: 20
  Correct Moves: 11 (55.00%)
  Valid Moves: 20 (100.00%)

gpt-3.5-turbo-legal:
  Total Puzzles: 20
  Correct Moves: 9 (45.00%)
  Valid Moves: 20 (100.00%)

================================================================================
```

## Move Validation

The script validates LLM outputs by:

1. Attempting to parse as UCI format (e.g., "e2e4")
2. Attempting to parse as SAN format (e.g., "Nf3")
3. Extracting potential moves from the response text
4. Checking if the move is legal in the current position

## Notes

- API rate limiting: The script includes delays to avoid hitting OpenAI rate limits
- Stockfish depth: Set to 20 for analysis (adjustable in code)
- The Lichess daily puzzle API may return the same puzzle if called too frequently
- For production testing with 100+ unique puzzles, consider downloading the full puzzle database

## Troubleshooting

**Stockfish not found:**
- Ensure the STOCKFISH_PATH in .env points to the correct executable
- Test by running the stockfish executable directly from command line

**OpenAI API errors:**
- Verify your API key is correct in the `.env` file
- Check you have sufficient API credits at [https://platform.openai.com/usage](https://platform.openai.com/usage)
- If you get a 429 error, you've exceeded your quota - add credits to your account
- Update model names in `puzzle_tester.py` if different models are available in your account

**CSV file not found:**
- Verify the LICHESS_PUZZLE_CSV path in your `.env` file
- Make sure you've downloaded and extracted the puzzle database
- The script will fall back to sample puzzles if CSV is not found

## License

MIT
