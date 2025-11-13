# Demo Scripts

This directory contains demo, debug, validation, and example scripts for the Azure-OpenAI Stock Analysis Tool project.

## Script Categories

### Demo Scripts
- `demo_mis_presentation.py` - MIS presentation demo
- `citation_demo.py` - Citation system demonstration
- `brave_quality_demo.py` - Brave search quality demo

### Debug Scripts
- `debug_search.py` - General search debugging
- `debug_search_flow.py` - Search flow debugging
- `debug_japanese_search.py` - Japanese search debugging
- `debug_sbi_bank_search.py` - SBI bank search debugging

### Validation Scripts
- `validate_enhancements.py` - Enhancement validation
- `validate_fix.py` - Fix validation
- `final_validation.py` - Final system validation

### Performance Benchmarks
- `performance_benchmark.py` - Performance benchmarking suite
- `performance_test.py` - Performance testing

### Example Integrations
- `example_simple_integration.py` - Simple integration example

## Usage

### Running Demo Scripts
```bash
# From project root
python demos/citation_demo.py
python demos/brave_quality_demo.py
python demos/demo_mis_presentation.py
```

### Running Debug Scripts
```bash
# Debug general search
python demos/debug_search.py

# Debug Japanese search
python demos/debug_japanese_search.py

# Debug search flow
python demos/debug_search_flow.py
```

### Running Validation Scripts
```bash
# Validate enhancements
python demos/validate_enhancements.py

# Run final validation
python demos/final_validation.py
```

### Running Performance Benchmarks
```bash
# Run performance benchmark
python demos/performance_benchmark.py

# Run performance test
python demos/performance_test.py
```

## Requirements

Most scripts require:
- Running backend server (`python main.py`)
- Valid API keys in `.env` file
- LocalStack for AWS-related demos (`docker-compose up -d`)

## Output

Demo scripts typically provide:
- Colored console output for readability
- Performance metrics and timing information
- Detailed results and comparisons
- Visual demonstrations of features

## Purpose

These scripts serve multiple purposes:
- **Demos**: Showcase features for presentations and stakeholders
- **Debug**: Troubleshoot specific issues during development
- **Validation**: Verify fixes and enhancements work correctly
- **Benchmarks**: Measure and optimize performance
- **Examples**: Provide integration examples for developers

## Contributing

When adding new demo/debug scripts:
1. Use descriptive filenames (`demo_*.py`, `debug_*.py`, `validate_*.py`)
2. Include docstrings explaining the script's purpose
3. Add proper error handling and informative output
4. Update this README with the new script
5. Keep scripts focused on a single purpose
