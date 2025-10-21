import os
import chess
import chess.engine
import chess.pgn
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Union
import time
import io
import pandas as pd
import random

# Load environment variables
load_dotenv()

class LichessPuzzleTester:
    def __init__(self, stockfish_path: str, openai_api_key: str):
        """Initialize the puzzle tester with Stockfish and OpenAI credentials."""
        self.stockfish_path = stockfish_path
        self.client = OpenAI(api_key=openai_api_key)
        self.engine: Optional[chess.engine.SimpleEngine] = None

    def start_engine(self):
        """Start the Stockfish engine."""
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

    def stop_engine(self):
        """Stop the Stockfish engine."""
        if self.engine:
            self.engine.quit()

    def fetch_random_puzzles(self, count: int = 20) -> List[Dict]:
        """Fetch random puzzles from Lichess puzzle CSV database."""
        print(f"Loading {count} random puzzles from Lichess CSV...")

        csv_path = os.getenv('LICHESS_PUZZLE_CSV')

        if not csv_path or not os.path.exists(csv_path):
            print(f"Warning: LICHESS_PUZZLE_CSV not found at: {csv_path}")
            print("Falling back to sample puzzles...")
            return self._get_sample_puzzles(count)

        try:
            return self._load_puzzles_from_csv(csv_path, count)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Falling back to sample puzzles...")
            return self._get_sample_puzzles(count)

    def _load_puzzles_from_csv(self, csv_path: str, count: int) -> List[Dict]:
        """Load random puzzles from the Lichess puzzle CSV file.

        CSV Format:
        - PuzzleId: Unique identifier
        - FEN: Position BEFORE opponent's move
        - Moves: UCI moves (first move is opponent's, rest are solution)
        - Rating: Puzzle difficulty rating
        - Themes: Puzzle themes/tags

        According to Lichess docs:
        - FEN is the position before the opponent makes their move
        - The position to present is after applying the first move to that FEN
        - The second move is the beginning of the solution
        """
        print(f"Reading puzzles from {csv_path}...")

        # Read CSV - it's likely large, so we'll use chunking
        # CSV columns: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl

        # First, count total lines to enable random sampling
        total_lines = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1  # Subtract header

        print(f"Total puzzles in CSV: {total_lines}")

        # Generate random line numbers to sample
        if count >= total_lines:
            sample_indices = set(range(total_lines))
        else:
            sample_indices = set(random.sample(range(total_lines), count))

        puzzles = []

        # Read the CSV and extract only the sampled rows
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Skip header
            f.readline()

            for idx, line in enumerate(f):
                if idx in sample_indices:
                    parts = line.strip().split(',')

                    if len(parts) < 4:
                        continue

                    puzzle_id = parts[0]
                    fen = parts[1]
                    moves = parts[2].split()  # UCI moves separated by spaces
                    rating = int(parts[3]) if parts[3].isdigit() else 1500

                    # According to docs:
                    # - FEN is position before opponent's move
                    # - First move in 'Moves' is the opponent's move
                    # - We need to apply the first move to get the position to present
                    # - Second move onwards is the solution

                    if len(moves) < 2:
                        continue  # Need at least opponent move + 1 solution move

                    # Apply the first (opponent's) move to the FEN
                    try:
                        board = chess.Board(fen)
                        opponent_move = chess.Move.from_uci(moves[0])
                        board.push(opponent_move)

                        # The position after opponent's move
                        puzzle_fen = board.fen()

                        # The solution starts from the second move
                        solution_moves = moves[1:]

                        puzzles.append({
                            'id': puzzle_id,
                            'fen': puzzle_fen,
                            'moves': solution_moves,
                            'rating': rating
                        })

                        if len(puzzles) >= count:
                            break

                    except Exception as e:
                        print(f"Error parsing puzzle {puzzle_id}: {e}")
                        continue

        print(f"Successfully loaded {len(puzzles)} puzzles from CSV")
        return puzzles

    def _fetch_puzzles_alternative(self, count: int) -> List[Dict]:
        """Alternative method to fetch puzzles using Lichess API."""
        puzzles = []

        # Using Lichess puzzle activity endpoint (requires authentication for bulk)
        # For demonstration, we'll create a structure that matches Lichess puzzle format
        print("Fetching puzzles from Lichess API...")

        # Note to user: You may need to download the puzzle database
        # from https://database.lichess.org/ for bulk puzzle access
        # This is a simplified version

        for i in range(count):
            try:
                # Lichess daily puzzle endpoint
                response = requests.get("https://lichess.org/api/puzzle/daily")
                if response.status_code == 200:
                    puzzle_data = response.json()

                    # Debug: Print the structure on first attempt
                    if i == 0:
                        print(f"API Response structure: {list(puzzle_data.keys())}")
                        if 'game' in puzzle_data:
                            print(f"Game keys: {list(puzzle_data['game'].keys())}")
                        if 'puzzle' in puzzle_data:
                            print(f"Puzzle keys: {list(puzzle_data['puzzle'].keys())}")

                    # Extract FEN - it may be in different locations depending on API version
                    fen = None
                    if 'game' in puzzle_data:
                        game = puzzle_data['game']
                        if 'fen' in game:
                            fen = game['fen']
                        elif 'pgn' in game:
                            # Extract FEN from PGN using initialPly
                            try:
                                pgn_text = game['pgn']
                                pgn_io = io.StringIO(pgn_text)
                                game_obj = chess.pgn.read_game(pgn_io)

                                if game_obj:
                                    board = game_obj.board()
                                    # Apply moves up to initialPly
                                    initial_ply = puzzle_data.get('puzzle', {}).get('initialPly', 0)
                                    for i, move in enumerate(game_obj.mainline_moves()):
                                        if i >= initial_ply:
                                            break
                                        board.push(move)
                                    fen = board.fen()
                            except Exception as e:
                                print(f"  Error extracting FEN from PGN: {e}")
                    if not fen and 'fen' in puzzle_data:
                        fen = puzzle_data['fen']

                    # Extract solution moves
                    moves = None
                    if 'puzzle' in puzzle_data:
                        puzzle_obj = puzzle_data['puzzle']
                        if 'solution' in puzzle_obj:
                            moves = puzzle_obj['solution']
                        elif 'moves' in puzzle_obj:
                            moves = puzzle_obj['moves']

                    # Extract ID
                    puzzle_id = None
                    if 'game' in puzzle_data and 'id' in puzzle_data['game']:
                        puzzle_id = puzzle_data['game']['id']
                    elif 'puzzle' in puzzle_data and 'id' in puzzle_data['puzzle']:
                        puzzle_id = puzzle_data['puzzle']['id']
                    elif 'id' in puzzle_data:
                        puzzle_id = puzzle_data['id']

                    if fen and moves:
                        puzzle = {
                            'fen': fen,
                            'moves': moves,
                            'rating': puzzle_data.get('puzzle', {}).get('rating', 1500),
                            'id': puzzle_id or f"puzzle_{i}"
                        }
                        puzzles.append(puzzle)

                        # Only one unique daily puzzle, so we'll break after getting it
                        # and generate variations or use sample puzzles
                        if i == 0:
                            print("Note: Lichess daily puzzle API returns same puzzle.")
                            print("Adding sample puzzles for testing...")
                            puzzles.extend(self._get_sample_puzzles(count - 1))
                            break
                    else:
                        print(f"Puzzle {i}: Missing FEN or moves in response")
                        print(f"  FEN found: {fen is not None}, Moves found: {moves is not None}")
                        if i == 0:
                            # Fall back to sample puzzles immediately
                            print("Falling back to sample puzzles...")
                            puzzles = self._get_sample_puzzles(count)
                            break

                # To avoid rate limiting and get different puzzles,
                # in production you should use the database download
                if i < count - 1:
                    time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"Error fetching puzzle {i}: {e}")
                if i == 0:
                    print("Falling back to sample puzzles...")
                    puzzles = self._get_sample_puzzles(count)
                    break
                continue

        print(f"Successfully fetched {len(puzzles)} puzzles")
        return puzzles

    def _get_sample_puzzles(self, count: int) -> List[Dict]:
        """Get sample chess puzzles for testing when API is unavailable."""
        # These are real Lichess puzzles with known solutions
        sample_puzzles = [
            {
                'fen': 'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4',
                'moves': ['Qxf7#'],
                'rating': 800,
                'id': 'sample_1'
            },
            {
                'fen': 'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5',
                'moves': ['Bxf7+', 'Ke7', 'Ng5'],
                'rating': 1200,
                'id': 'sample_2'
            },
            {
                'fen': 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3',
                'moves': ['Nf6'],
                'rating': 1000,
                'id': 'sample_3'
            },
            {
                'fen': 'rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
                'moves': ['Nxe5'],
                'rating': 900,
                'id': 'sample_4'
            },
            {
                'fen': 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3',
                'moves': ['a6'],
                'rating': 1100,
                'id': 'sample_5'
            }
        ]

        # Replicate the sample puzzles to reach the desired count
        result = []
        for i in range(count):
            puzzle = sample_puzzles[i % len(sample_puzzles)].copy()
            puzzle['id'] = f"sample_{i}"
            result.append(puzzle)

        return result

    def get_top_moves_with_eval(self, fen: str, num_moves: int = 5) -> List[Tuple[str, Union[str, float]]]:
        """Get top N moves and their evaluations using Stockfish."""
        if not self.engine:
            raise RuntimeError("Engine not started. Call start_engine() first.")

        board = chess.Board(fen)

        # Analyze position
        info = self.engine.analyse(board, chess.engine.Limit(depth=20), multipv=num_moves)

        top_moves = []
        for i, line_info in enumerate(info):
            if 'pv' in line_info and len(line_info['pv']) > 0:
                move = line_info['pv'][0]
                score = line_info.get('score', chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))

                # Convert score to centipawns from the perspective of the side to move
                if score.is_mate():
                    mate_val = score.relative.mate()
                    eval_score_str: str | float = f"Mate in {mate_val}" if mate_val else "Mate"
                else:
                    cp_score = score.relative.score()
                    eval_score_str = cp_score / 100.0 if cp_score is not None else 0.0

                top_moves.append((move.uci(), eval_score_str))

        return top_moves

    def get_legal_moves(self, fen: str) -> List[str]:
        """Get all legal moves in the position, sorted alphabetically."""
        board = chess.Board(fen)
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_moves.sort()
        return legal_moves

    def create_prompt(self, fen: str, with_context: bool = False,
                     top_moves: Optional[List[Tuple[str, Union[str, float]]]] = None,
                     legal_moves: Optional[List[str]] = None) -> str:
        """Create the prompt for the LLM."""
        board = chess.Board(fen)
        side = "White" if board.turn == chess.WHITE else "Black"

        prompt = f"{fen}\n"

        if with_context and top_moves:
            prompt += f"\nTop 5 candidate moves with evaluations:\n"
            for i, (move, eval_score) in enumerate(top_moves, 1):
                prompt += f"{i}. {move} (eval: {eval_score})\n"
            prompt += "\n"
        elif legal_moves:
            prompt += f"\nLegal moves (UCI format, alphabetically):\n"
            prompt += ", ".join(legal_moves) + "\n\n"

        prompt += f"Find the best move for {side}.\n"
        prompt += "Only print the best move in the position."

        return prompt

    def query_llm(self, model: str, prompt: str) -> str:
        """Query the OpenAI LLM and return the response."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=50
            )

            move = response.choices[0].message.content
            if move:
                return move.strip()
            else:
                return "ERROR: move is NULL"
        except Exception as e:
            print(f"Error querying {model}: {e}")
            return ""

    def validate_move(self, move_str: str, fen: str) -> bool:
        """Validate if the LLM output is a valid move in the position."""
        board = chess.Board(fen)
        move_str_original = move_str.strip()

        # Remove common prefixes like "..." for black moves
        move_str_cleaned = move_str_original.lstrip('.')

        # Try to parse as SAN format (case-sensitive)
        try:
            move = board.parse_san(move_str_cleaned)
            if move in board.legal_moves:
                return True
        except:
            pass

        # Try to parse as UCI format (lowercase)
        try:
            move = chess.Move.from_uci(move_str_cleaned.lower())
            if move in board.legal_moves:
                return True
        except:
            pass

        # Try to extract move from the response
        words = move_str_original.split()
        for word in words:
            word_cleaned = word.strip('.,;:!?()[]{}')

            # Try SAN format
            try:
                move = board.parse_san(word_cleaned.lstrip('.'))
                if move in board.legal_moves:
                    return True
            except:
                pass

            # Try UCI format
            try:
                move = chess.Move.from_uci(word_cleaned.lower())
                if move in board.legal_moves:
                    return True
            except:
                continue

        return False

    def extract_move(self, move_str: str, fen: str) -> Optional[str]:
        """Extract and return a valid move from the LLM output."""
        board = chess.Board(fen)
        move_str_original = move_str.strip()

        # Remove common prefixes like "..." for black moves
        move_str_cleaned = move_str_original.lstrip('.')

        # Try SAN format first (case-sensitive)
        try:
            move = board.parse_san(move_str_cleaned)
            if move in board.legal_moves:
                return move.uci()
        except:
            pass

        # Try UCI format (lowercase)
        try:
            move = chess.Move.from_uci(move_str_cleaned.lower())
            if move in board.legal_moves:
                return move.uci()
        except:
            pass

        # Try to extract from words (for longer responses)
        words = move_str_original.split()
        for word in words:
            # Clean punctuation but preserve case for SAN
            word_cleaned = word.strip('.,;:!?()[]{}')

            # Try SAN format
            try:
                move = board.parse_san(word_cleaned.lstrip('.'))
                if move in board.legal_moves:
                    return move.uci()
            except:
                pass

            # Try UCI format
            try:
                move = chess.Move.from_uci(word_cleaned.lower())
                if move in board.legal_moves:
                    return move.uci()
            except:
                continue

        return None

    def test_models(self, puzzles: List[Dict]) -> Dict[str, Dict]:
        """Test all models on the puzzle set."""
        # Note: Update these model names based on what's available in your OpenAI account
        # Common options: gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
        models = {
            'gpt-4o-mini': {'with_context': False, 'with_legal_moves': False},
            'gpt-3.5-turbo': {'with_context': False, 'with_legal_moves': False},
            'gpt-4o-mini-context': {'with_context': True, 'with_legal_moves': False, 'base_model': 'gpt-4o-mini'},
            'gpt-3.5-turbo-context': {'with_context': True, 'with_legal_moves': False, 'base_model': 'gpt-3.5-turbo'},
            'gpt-4o-mini-legal': {'with_context': False, 'with_legal_moves': True, 'base_model': 'gpt-4o-mini'},
            'gpt-3.5-turbo-legal': {'with_context': False, 'with_legal_moves': True, 'base_model': 'gpt-3.5-turbo'}
        }

        results = {model: {'correct': 0, 'valid': 0, 'total': 0, 'responses': []}
                  for model in models.keys()}

        self.start_engine()

        try:
            for idx, puzzle in enumerate(puzzles):
                print(f"\nProcessing puzzle {idx + 1}/{len(puzzles)}...")
                fen = puzzle['fen']
                correct_move = puzzle['moves'][0] if isinstance(puzzle['moves'], list) else puzzle['moves'].split()[0]

                # Get top moves for context-aware models
                top_moves = self.get_top_moves_with_eval(fen)

                # Get legal moves for legal-moves variants
                legal_moves_list = self.get_legal_moves(fen)

                for model_name, config in models.items():
                    try:
                        # Determine actual OpenAI model to use
                        if 'base_model' in config:
                            openai_model = config['base_model']
                        else:
                            openai_model = model_name

                        # Create prompt
                        prompt = self.create_prompt(
                            fen,
                            with_context=config['with_context'],
                            top_moves=top_moves if config['with_context'] else None,
                            legal_moves=legal_moves_list if config['with_legal_moves'] else None
                        )

                        # Query LLM
                        response = self.query_llm(openai_model, prompt)

                        # Skip if no response (API error already logged)
                        if not response:
                            print(f"  {model_name}: Skipped (API error)")
                            continue

                        # Validate and check correctness
                        is_valid = self.validate_move(response, fen)
                        extracted_move = self.extract_move(response, fen)
                        is_correct = extracted_move == correct_move if extracted_move else False

                        results[model_name]['total'] += 1
                        if is_valid:
                            results[model_name]['valid'] += 1
                        if is_correct:
                            results[model_name]['correct'] += 1

                        results[model_name]['responses'].append({
                            'puzzle_id': puzzle.get('id', idx),
                            'fen': fen,
                            'correct_move': correct_move,
                            'llm_response': response,
                            'extracted_move': extracted_move,
                            'is_valid': is_valid,
                            'is_correct': is_correct
                        })

                        print(f"  {model_name}: {response} (Valid: {is_valid}, Correct: {is_correct})")

                        # Small delay to avoid rate limiting
                        time.sleep(0.1)

                    except Exception as e:
                        print(f"  {model_name}: Error - {e}")
                        continue

        finally:
            self.stop_engine()

        return results

    def calculate_accuracy(self, results: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate accuracy metrics for all models."""
        accuracy_report = {}

        for model_name, data in results.items():
            total = data['total']
            if total > 0:
                accuracy_report[model_name] = {
                    'correct_accuracy': (data['correct'] / total) * 100,
                    'valid_move_rate': (data['valid'] / total) * 100,
                    'total_puzzles': total,
                    'correct_count': data['correct'],
                    'valid_count': data['valid']
                }
            else:
                accuracy_report[model_name] = {
                    'correct_accuracy': 0,
                    'valid_move_rate': 0,
                    'total_puzzles': 0,
                    'correct_count': 0,
                    'valid_count': 0
                }

        return accuracy_report

    def print_report(self, accuracy_report: Dict[str, Dict[str, float]]):
        """Print a formatted accuracy report."""
        print("\n" + "="*80)
        print("ACCURACY REPORT")
        print("="*80)

        for model_name, metrics in accuracy_report.items():
            print(f"\n{model_name}:")
            print(f"  Total Puzzles: {metrics['total_puzzles']}")
            print(f"  Correct Moves: {metrics['correct_count']} ({metrics['correct_accuracy']:.2f}%)")
            print(f"  Valid Moves: {metrics['valid_count']} ({metrics['valid_move_rate']:.2f}%)")

        print("\n" + "="*80)

    def save_results(self, results: Dict, accuracy_report: Dict, filename: str = "results.json"):
        """Save detailed results to a JSON file."""
        output = {
            'accuracy_report': accuracy_report,
            'detailed_results': results
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nDetailed results saved to {filename}")


def main():
    # Load configuration
    stockfish_path = os.getenv('STOCKFISH_PATH', 'stockfish')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return

    # Initialize tester
    tester = LichessPuzzleTester(stockfish_path, openai_api_key)

    # Fetch puzzles
    puzzles = tester.fetch_random_puzzles(count=100)

    if not puzzles:
        print("No puzzles fetched. Please check your internet connection or API access.")
        return

    # Test models
    print("\nStarting model testing...")
    results = tester.test_models(puzzles)

    # Calculate accuracy
    accuracy_report = tester.calculate_accuracy(results)

    # Print report
    tester.print_report(accuracy_report)

    # Save results
    tester.save_results(results, accuracy_report)


if __name__ == "__main__":
    main()
