# LLM Adaptive Computation

Note this is not fully up to data

A project for evaluating LLaMA models on mathematical reasoning tasks using the GSM8K dataset.

## Overview

This project implements a mathematical reasoning system using TinyLlama (1.1B parameters) optimized for CPU usage. It evaluates the model's performance on grade-school math problems from the GSM8K dataset.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use the adaptive computation model, run the following command:

```bash
python src/run.py --log-file logs/custom_eval.log --num-samples 5
```

For more detailed usage instructions, refer to the [documentation](docs/USAGE.md).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
