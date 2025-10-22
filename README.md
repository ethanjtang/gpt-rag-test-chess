# Lichess Puzzle Tester for GPT Models

This project tests different GPT mini models on Lichess chess puzzles to evaluate their chess-playing capabilities. A great example of how simple retrieval-augmented generation leads to such large improvements in sanity and accuracy for LLMs.

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

You are free to steal this code - it is not mine in any sense of the world!
