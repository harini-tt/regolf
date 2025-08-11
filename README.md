# Regex Golf GRPO Training System

## Overview

Regex golf — the challenge of finding the shortest possible regex that matches all positives and rejects all negatives — is an NP-hard problem. This repository implements a reinforcement learning approach using Group Relative Policy Optimization (GRPO) to generate high-quality approximate solutions for regex golf puzzles.

While directly tackling regex golf, this framework is generalizable to a wide class of NP-hard problems such as scheduling, routing, and combinatorial optimization, making it a potential blueprint for solving other computationally difficult tasks.

## Societal Impact

Regular expressions power critical systems across industries:
- **Content Moderation**: Filtering hate speech and harmful content on social platforms
- **Cybersecurity**: Detecting malware signatures and threat patterns
- **Fraud Prevention**: Identifying scam patterns in financial transactions
- **Healthcare**: Validating and cleaning medical data pipelines
- **Data Quality**: Ensuring data integrity in large-scale processing systems

Our solver enables faster, safer, and more reliable text-matching systems at scale.

## Technical Architecture

### Core Components

- **AST Validation** (`regex/regex_ast_validation.py`): Prevents regex bombs through static analysis
- **Model Parser** (`regex/model_output_parser.py`): Parses Harmony v2 format from model outputs
- **Reward Calculator** (`regex/reward_calculator.py`): Computes rewards based on regex performance
- **Data Validation** (`data/validation/`): Ensures training data quality
- **GRPO Training** (pending): Fine-tuning with Group Relative Policy Optimization

### Key Features

- Grammar-constrained decoding for valid regex generation
- Curriculum difficulty scaling for stable learning
- Catastrophic backtracking prevention with timeout mechanisms
- Multiprocessing-based validation for handling complex patterns
- Support for 120B parameter models

## Dataset

Our training set consists of diverse puzzles generated using GPT-5, including:
- Creative adversarial patterns
- Public forum content moderation scenarios
- Multilingual text parsing challenges
- Cybersecurity threat detection patterns
- Spam filtering examples
- Log analysis patterns
- Large-scale data validation scenarios

All puzzles are validated through deterministic scripts to ensure:
- Non-trivial difficulty
- Correctness of YES/NO classifications
- No degenerate cases

### Dataset Statistics
- Main dataset: 4,303 entries
- Extended dataset: 7,357 entries (from checkpoints)
- ~100 test strings per entry
- 59% complex regex patterns

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/regolf.git
cd regolf

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Validation

Validate your regex datasets to ensure quality:

```bash
cd data/validation
python data_validator_fast.py
```

This uses multiprocessing with hard timeouts to handle patterns that cause catastrophic backtracking.

### Training (Coming Soon)

```bash
python train_grpo.py \
  --model_size 120B \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --num_epochs 10
```

## Project Structure

```
regolf/
├── regex/                          # Core regex processing modules
│   ├── regex_ast_validation.py    # AST validation for regex safety
│   ├── model_output_parser.py     # Harmony v2 format parser
│   ├── reward_calculator.py       # Reward computation
│   └── regex_evaluator.py         # Main integration module
├── data/
│   ├── generated_data/            # Training datasets
│   ├── validated_data/            # Cleaned datasets
│   ├── generation/                # Dataset generation scripts
│   └── validation/                # Validation scripts
│       ├── data_validator.py
│       ├── data_validator_fast.py # Multiprocessing validator
│       └── data_validator_entry_progress.py
├── prompts/
│   └── agent_prompts/             # System and agent prompts
│       ├── developer_message.py
│       ├── system_message.py
│       └── user_message.py
├── tests/
│   └── regex/tests/               # Unit tests
├── requirements.txt
└── README.md
```

## Technical Details

### Validation Pipeline

1. **AST Analysis**: Static analysis prevents exponential-time patterns
2. **Timeout Protection**: Multiprocessing with hard kills for stuck patterns
3. **Full Match Validation**: Uses `re.fullmatch()` for exact string matching
4. **Progress Tracking**: Entry-level progress bars for large datasets

### Reward Shaping

The reward function balances:
- Regex length (shorter is better)
- Accuracy on YES strings
- Accuracy on NO strings
- Compilation success
- Execution time constraints

## Performance

- Validation speed: ~100-200 entries/second
- Handles patterns with 100+ test strings
- Hard timeout: 1 second per pattern
- Memory efficient: Processes files sequentially

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Future Work

- [ ] Complete GRPO training implementation
- [ ] Add Modal infrastructure for multi-GPU support
- [ ] Integrate Wandb logging
- [ ] Implement beam search for regex generation
- [ ] Add support for extended regex features
- [ ] Create interactive demo interface

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenAI for the 120B OSS model
- GPT-5 for dataset generation
- The regex golf community for inspiration

## Citation

If you use this work in your research, please cite:

```bibtex
@software{regolf_grpo,
  title = {Regex Golf GRPO Training System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/regolf}
}
```