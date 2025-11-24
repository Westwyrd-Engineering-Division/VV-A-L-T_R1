# V.V.A.L.T Command Line Interface (CLI) Usage Guide

The V.V.A.L.T CLI provides a comprehensive command-line interface for the Vantage-Vector Autonomous Logic Transformer, enabling easy access to inference, verification, configuration, and management tools.

## Installation

After installing the V.V.A.L.T package, the `vvalt` command will be available in your terminal:

```bash
pip install -e .
```

Or from PyPI (when published):

```bash
pip install vvalt
```

## Quick Start

View available commands:

```bash
vvalt --help
```

Run a basic demo:

```bash
vvalt demo --example basic
```

Check version and system info:

```bash
vvalt info
```

## Commands Overview

The CLI is organized into the following main commands:

- **`infer`** - Run inference on input data
- **`explain`** - Generate human-readable reasoning traces
- **`verify`** - Verify model safety and determinism
- **`config`** - Manage configurations (generate, validate, show)
- **`checkpoint`** - Manage model checkpoints (create, load, info)
- **`demo`** - Run built-in demonstrations
- **`info`** - Display system information

---

## Command Reference

### 1. Inference (`vvalt infer`)

Run V.V.A.L.T inference on input data with multi-perspective reasoning.

#### Basic Usage

```bash
vvalt infer --input data.json --task task.json
```

#### Options

| Option | Description |
|--------|-------------|
| `-i, --input FILE` | Input data file (JSON or NPY format) **[required]** |
| `-t, --task FILE` | Task specification file (JSON or YAML) |
| `-c, --config FILE` | Configuration file (YAML) |
| `-o, --output FILE` | Output file for results (JSON or NPY) |
| `--checkpoint FILE` | Load model from checkpoint file |
| `--enhanced` | Use enhanced VVALT with monitoring |
| `--batch-size N` | Batch size for processing (default: auto) |
| `--format {json,npy,text}` | Output format (default: json) |

#### Examples

Run inference with default configuration:
```bash
vvalt infer --input my_data.json --output results.json
```

Run with custom configuration and task:
```bash
vvalt infer \
  --input data.npy \
  --task task.yaml \
  --config my_config.yaml \
  --output results.npy \
  --format npy
```

Use enhanced model with checkpoint:
```bash
vvalt infer \
  --input data.json \
  --checkpoint model.ckpt \
  --enhanced \
  --batch-size 32
```

---

### 2. Explanation (`vvalt explain`)

Generate detailed, human-readable reasoning traces showing how V.V.A.L.T arrived at its conclusions.

#### Basic Usage

```bash
vvalt explain --input data.json
```

#### Options

| Option | Description |
|--------|-------------|
| `-i, --input FILE` | Input data file (JSON or NPY format) **[required]** |
| `-t, --task FILE` | Task specification file (JSON or YAML) |
| `-c, --config FILE` | Configuration file (YAML) |
| `-o, --output FILE` | Output file for explanation (TXT or JSON) |
| `--checkpoint FILE` | Load model from checkpoint file |
| `--verbose` | Include detailed component traces |

#### Examples

Get basic explanation:
```bash
vvalt explain --input data.json
```

Save detailed explanation to file:
```bash
vvalt explain \
  --input data.json \
  --task task.yaml \
  --output explanation.txt \
  --verbose
```

---

### 3. Verification (`vvalt verify`)

Verify model safety guarantees and deterministic behavior.

#### Basic Usage

```bash
vvalt verify --samples 100
```

#### Options

| Option | Description |
|--------|-------------|
| `-c, --config FILE` | Configuration file (YAML) |
| `--checkpoint FILE` | Load model from checkpoint file |
| `--samples N` | Number of random samples for testing (default: 100) |
| `--determinism` | Run determinism verification |
| `--safety` | Generate safety report |
| `--all` | Run all verification checks |
| `-o, --output FILE` | Output file for verification report (JSON) |

#### Examples

Run all verification checks:
```bash
vvalt verify --all --samples 1000
```

Verify determinism only:
```bash
vvalt verify --determinism --samples 500
```

Generate safety report and save:
```bash
vvalt verify --safety --output safety_report.json
```

---

### 4. Configuration (`vvalt config`)

Manage V.V.A.L.T configuration files.

#### Subcommands

##### 4.1. Generate Configuration (`vvalt config generate`)

Create a new configuration file with presets.

```bash
vvalt config generate --output config.yaml --preset production
```

**Presets:**
- `default` - Standard configuration
- `fast` - Optimized for speed (smaller model)
- `accurate` - Optimized for accuracy (larger model)
- `production` - Production-ready with monitoring

**Examples:**

```bash
# Generate default config
vvalt config generate

# Generate production config
vvalt config generate --output prod_config.yaml --preset production

# Generate fast config for testing
vvalt config generate --output fast_config.yaml --preset fast
```

##### 4.2. Validate Configuration (`vvalt config validate`)

Validate a configuration file.

```bash
vvalt config validate config.yaml
```

##### 4.3. Show Configuration (`vvalt config show`)

Display configuration file contents.

```bash
vvalt config show config.yaml --format json
```

---

### 5. Checkpoint Management (`vvalt checkpoint`)

Create, load, and inspect model checkpoints.

#### Subcommands

##### 5.1. Create Checkpoint (`vvalt checkpoint create`)

Create a new model checkpoint.

```bash
vvalt checkpoint create --config config.yaml --output model.ckpt
```

**Options:**
- `-c, --config FILE` - Configuration file (YAML)
- `-o, --output FILE` - Output checkpoint file **[required]**
- `--metadata FILE` - Metadata JSON file to include

**Examples:**

```bash
# Create checkpoint with default config
vvalt checkpoint create --output checkpoint.ckpt

# Create with custom config and metadata
vvalt checkpoint create \
  --config my_config.yaml \
  --output trained_model.ckpt \
  --metadata model_info.json
```

##### 5.2. Load Checkpoint (`vvalt checkpoint load`)

Load and optionally verify a checkpoint.

```bash
vvalt checkpoint load checkpoint.ckpt --verify
```

##### 5.3. Checkpoint Info (`vvalt checkpoint info`)

Display checkpoint information.

```bash
vvalt checkpoint info checkpoint.ckpt --verbose
```

---

### 6. Demonstrations (`vvalt demo`)

Run built-in demonstrations of V.V.A.L.T capabilities.

#### Basic Usage

```bash
vvalt demo --example basic
```

#### Examples

| Example | Description |
|---------|-------------|
| `basic` | Simple inference demonstration |
| `production` | Enhanced features and monitoring |
| `perspectives` | Multi-perspective analysis |
| `safety` | Safety verification and guarantees |
| `all` | Run all demonstrations |

#### Examples

```bash
# Run basic demo
vvalt demo --example basic

# Run all demos with verbose output
vvalt demo --example all --verbose

# Run with custom config
vvalt demo --example production --config my_config.yaml
```

---

### 7. System Information (`vvalt info`)

Display V.V.A.L.T system information and version details.

#### Basic Usage

```bash
vvalt info
```

#### Options

| Option | Description |
|--------|-------------|
| `--verbose` | Show detailed information |
| `--check-deps` | Check optional dependencies |

#### Examples

```bash
# Basic info
vvalt info

# Detailed info with dependency check
vvalt info --verbose --check-deps
```

---

## Workflow Examples

### Complete Inference Workflow

```bash
# 1. Generate configuration
vvalt config generate --output config.yaml --preset production

# 2. Validate configuration
vvalt config validate config.yaml

# 3. Create checkpoint
vvalt checkpoint create --config config.yaml --output model.ckpt

# 4. Verify model
vvalt verify --checkpoint model.ckpt --all --samples 1000

# 5. Run inference
vvalt infer \
  --input data.json \
  --task task.yaml \
  --checkpoint model.ckpt \
  --output results.json

# 6. Get explanation
vvalt explain \
  --input data.json \
  --checkpoint model.ckpt \
  --output explanation.txt \
  --verbose
```

### Quick Testing Workflow

```bash
# Run demo to verify installation
vvalt demo --example basic

# Check system info
vvalt info --check-deps

# Run quick verification
vvalt verify --determinism --samples 50
```

### Configuration Development Workflow

```bash
# Generate different configs for experimentation
vvalt config generate --output fast.yaml --preset fast
vvalt config generate --output accurate.yaml --preset accurate
vvalt config generate --output prod.yaml --preset production

# Validate all configs
vvalt config validate fast.yaml
vvalt config validate accurate.yaml
vvalt config validate prod.yaml

# Compare configurations
vvalt config show fast.yaml --format json > fast.json
vvalt config show accurate.yaml --format json > accurate.json
diff fast.json accurate.json
```

---

## Input/Output Formats

### Input Data Formats

**JSON Format** (`.json`):
```json
[
  [0.1, 0.2, 0.3, ...],
  [0.4, 0.5, 0.6, ...]
]
```

**NumPy Format** (`.npy`):
```python
import numpy as np
data = np.random.randn(batch_size, input_dim).astype(np.float32)
np.save('data.npy', data)
```

### Task Specification Formats

**JSON Format**:
```json
{
  "task_type": "classification",
  "priorities": {
    "semantic": 0.3,
    "structural": 0.2,
    "causal": 0.2,
    "relational": 0.15,
    "graph": 0.15
  }
}
```

**YAML Format**:
```yaml
task_type: classification
priorities:
  semantic: 0.3
  structural: 0.2
  causal: 0.2
  relational: 0.15
  graph: 0.15
```

### Configuration Format

**YAML Configuration**:
```yaml
model:
  input_dim: 512
  hidden_dim: 512
  num_heads: 8
  num_perspectives: 5

runtime:
  enable_monitoring: true
  enable_validation: true
  deterministic: true

performance:
  enable_caching: true
  batch_size: 32

logging:
  level: "INFO"
  file: "vvalt.log"
```

---

## Environment Variables

The CLI supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `VVALT_CONFIG` | Default configuration file path | None |
| `VVALT_CHECKPOINT_DIR` | Default checkpoint directory | `./checkpoints` |
| `VVALT_LOG_LEVEL` | Logging level | `INFO` |
| `VVALT_CACHE_DIR` | Cache directory | `~/.cache/vvalt` |

Example:
```bash
export VVALT_CONFIG=~/my_config.yaml
export VVALT_LOG_LEVEL=DEBUG
vvalt infer --input data.json
```

---

## Troubleshooting

### Common Issues

**1. Command not found: vvalt**

Solution: Reinstall the package in development mode
```bash
pip install -e .
```

**2. Import errors for optional dependencies**

Solution: Install with extras
```bash
pip install -e ".[pytorch,transformers]"
```

**3. Configuration validation errors**

Solution: Generate a fresh config and compare
```bash
vvalt config generate --output new_config.yaml
vvalt config show new_config.yaml
```

**4. Checkpoint loading failures**

Solution: Check checkpoint info
```bash
vvalt checkpoint info checkpoint.ckpt --verbose
```

---

## Advanced Usage

### Scripting with the CLI

```bash
#!/bin/bash
# Batch processing script

for data_file in data/*.json; do
  output_file="results/$(basename $data_file)"
  echo "Processing: $data_file"

  vvalt infer \
    --input "$data_file" \
    --config config.yaml \
    --output "$output_file" \
    --format json
done

echo "Batch processing complete"
```

### Pipeline Integration

```bash
# Use in a data pipeline
cat input.json | \
  python preprocess.py | \
  vvalt infer --input - --output - | \
  python postprocess.py > final_results.json
```

---

## API vs CLI

When to use the CLI vs the Python API:

**Use CLI when:**
- Running one-off inference tasks
- Quick testing and validation
- Configuration management
- CI/CD pipelines
- Scripting and automation
- Interactive exploration

**Use Python API when:**
- Integrating into applications
- Custom preprocessing/postprocessing
- Advanced model customization
- Real-time inference services
- Complex workflows
- Performance-critical applications

---

## Support

For issues, questions, or contributions:

- **Issues**: https://github.com/VValtDisney/V.V.A.L.T/issues
- **Documentation**: https://github.com/VValtDisney/V.V.A.L.T
- **License**: MIT (2025)

---

## Version History

### v1.0.0 (2025)
- Initial CLI release
- Core commands: infer, explain, verify
- Configuration management
- Checkpoint support
- Built-in demonstrations
- Comprehensive documentation
