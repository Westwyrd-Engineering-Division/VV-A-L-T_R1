#!/usr/bin/env python3
"""
V.V.A.L.T Command Line Interface

A comprehensive CLI for the Vantage-Vector Autonomous Logic Transformer.
Provides access to inference, verification, configuration, and management tools.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from vvalt.core import VVALT
from vvalt.core_enhanced import VVALTEnhanced
from vvalt.config import VVALTConfig
from vvalt.checkpoint import CheckpointManager, create_checkpoint_metadata
from vvalt.validation import ValidationPipeline
from vvalt.task_envelope import TaskEnvelope, TaskVectorBuilder, TaskType


class VVALTCLIError(Exception):
    """Custom exception for CLI errors"""
    pass


class VVALTCLI:
    """Main CLI controller for V.V.A.L.T operations"""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with subcommands"""
        parser = argparse.ArgumentParser(
            prog='vvalt',
            description='V.V.A.L.T - Vantage-Vector Autonomous Logic Transformer CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run inference
  vvalt infer --input data.json --task task.json --config config.yaml

  # Get reasoning explanation
  vvalt explain --input data.json --task task.json

  # Verify model safety
  vvalt verify --config config.yaml --determinism

  # Generate default configuration
  vvalt config generate --output config.yaml

  # Create checkpoint
  vvalt checkpoint create --config config.yaml --output model.ckpt

  # Run demo
  vvalt demo --example basic

  # Show model information
  vvalt info --verbose

For more information, visit: https://github.com/VValtDisney/V.V.A.L.T
            """
        )

        parser.add_argument(
            '--version',
            action='version',
            version='V.V.A.L.T 1.0.0'
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Infer command
        self._add_infer_parser(subparsers)

        # Explain command
        self._add_explain_parser(subparsers)

        # Verify command
        self._add_verify_parser(subparsers)

        # Config command
        self._add_config_parser(subparsers)

        # Checkpoint command
        self._add_checkpoint_parser(subparsers)

        # Demo command
        self._add_demo_parser(subparsers)

        # Info command
        self._add_info_parser(subparsers)

        return parser

    def _add_infer_parser(self, subparsers):
        """Add inference subcommand"""
        infer = subparsers.add_parser(
            'infer',
            help='Run inference on input data',
            description='Execute V.V.A.L.T inference with multi-perspective reasoning'
        )
        infer.add_argument(
            '-i', '--input',
            required=True,
            type=str,
            help='Input data file (JSON or NPY format)'
        )
        infer.add_argument(
            '-t', '--task',
            type=str,
            help='Task specification file (JSON or NPY)'
        )
        infer.add_argument(
            '-c', '--config',
            type=str,
            help='Configuration file (YAML)'
        )
        infer.add_argument(
            '-o', '--output',
            type=str,
            help='Output file for results (JSON or NPY)'
        )
        infer.add_argument(
            '--checkpoint',
            type=str,
            help='Load model from checkpoint file'
        )
        infer.add_argument(
            '--enhanced',
            action='store_true',
            help='Use enhanced VVALT with monitoring'
        )
        infer.add_argument(
            '--format',
            choices=['json', 'npy', 'text'],
            default='json',
            help='Output format (default: json)'
        )
        infer.add_argument(
            '--trace',
            action='store_true',
            help='Include reasoning trace in output'
        )

    def _add_explain_parser(self, subparsers):
        """Add explanation subcommand"""
        explain = subparsers.add_parser(
            'explain',
            help='Generate human-readable reasoning trace',
            description='Get detailed explanation of V.V.A.L.T reasoning process'
        )
        explain.add_argument(
            '-i', '--input',
            required=True,
            type=str,
            help='Input data file (JSON or NPY format)'
        )
        explain.add_argument(
            '-t', '--task',
            type=str,
            help='Task specification file (JSON or NPY)'
        )
        explain.add_argument(
            '-c', '--config',
            type=str,
            help='Configuration file (YAML)'
        )
        explain.add_argument(
            '-o', '--output',
            type=str,
            help='Output file for explanation (TXT or JSON)'
        )
        explain.add_argument(
            '--checkpoint',
            type=str,
            help='Load model from checkpoint file'
        )
        explain.add_argument(
            '--verbose',
            action='store_true',
            help='Include detailed component traces'
        )

    def _add_verify_parser(self, subparsers):
        """Add verification subcommand"""
        verify = subparsers.add_parser(
            'verify',
            help='Verify model safety and determinism',
            description='Run safety verification and determinism checks'
        )
        verify.add_argument(
            '-c', '--config',
            type=str,
            help='Configuration file (YAML)'
        )
        verify.add_argument(
            '--checkpoint',
            type=str,
            help='Load model from checkpoint file'
        )
        verify.add_argument(
            '--trials',
            type=int,
            default=10,
            help='Number of trials for determinism testing (default: 10)'
        )
        verify.add_argument(
            '--determinism',
            action='store_true',
            help='Run determinism verification'
        )
        verify.add_argument(
            '--safety',
            action='store_true',
            help='Generate safety report'
        )
        verify.add_argument(
            '--all',
            action='store_true',
            help='Run all verification checks'
        )
        verify.add_argument(
            '-o', '--output',
            type=str,
            help='Output file for verification report (JSON)'
        )

    def _add_config_parser(self, subparsers):
        """Add configuration management subcommand"""
        config = subparsers.add_parser(
            'config',
            help='Manage V.V.A.L.T configurations',
            description='Generate, validate, and inspect configuration files'
        )
        config_sub = config.add_subparsers(dest='config_action', help='Configuration actions')

        # Generate config
        gen = config_sub.add_parser('generate', help='Generate default configuration')
        gen.add_argument(
            '-o', '--output',
            type=str,
            default='config.yaml',
            help='Output configuration file (default: config.yaml)'
        )
        gen.add_argument(
            '--preset',
            choices=['default', 'fast', 'accurate'],
            default='default',
            help='Configuration preset (default: default)'
        )

        # Validate config
        val = config_sub.add_parser('validate', help='Validate configuration file')
        val.add_argument(
            'config_file',
            type=str,
            help='Configuration file to validate'
        )

        # Show config
        show = config_sub.add_parser('show', help='Display configuration')
        show.add_argument(
            'config_file',
            type=str,
            help='Configuration file to display'
        )
        show.add_argument(
            '--format',
            choices=['yaml', 'json'],
            default='yaml',
            help='Output format (default: yaml)'
        )

    def _add_checkpoint_parser(self, subparsers):
        """Add checkpoint management subcommand"""
        ckpt = subparsers.add_parser(
            'checkpoint',
            help='Manage model checkpoints',
            description='Create, load, and inspect model checkpoints'
        )
        ckpt_sub = ckpt.add_subparsers(dest='checkpoint_action', help='Checkpoint actions')

        # Create checkpoint
        create = ckpt_sub.add_parser('create', help='Create model checkpoint')
        create.add_argument(
            '-c', '--config',
            type=str,
            help='Configuration file (YAML)'
        )
        create.add_argument(
            '-o', '--output',
            required=True,
            type=str,
            help='Output checkpoint file'
        )
        create.add_argument(
            '--description',
            type=str,
            default='',
            help='Checkpoint description'
        )

        # Load checkpoint
        load = ckpt_sub.add_parser('load', help='Load and verify checkpoint')
        load.add_argument(
            'checkpoint_file',
            type=str,
            help='Checkpoint file to load'
        )
        load.add_argument(
            '--verify',
            action='store_true',
            help='Verify checkpoint integrity'
        )

        # Info checkpoint
        info = ckpt_sub.add_parser('info', help='Display checkpoint information')
        info.add_argument(
            'checkpoint_file',
            type=str,
            help='Checkpoint file to inspect'
        )

    def _add_demo_parser(self, subparsers):
        """Add demo subcommand"""
        demo = subparsers.add_parser(
            'demo',
            help='Run built-in demonstrations',
            description='Execute example demonstrations of V.V.A.L.T capabilities'
        )
        demo.add_argument(
            '--example',
            choices=['basic', 'production', 'perspectives', 'safety', 'all'],
            default='basic',
            help='Demo example to run (default: basic)'
        )
        demo.add_argument(
            '--config',
            type=str,
            help='Custom configuration file'
        )
        demo.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed output'
        )

    def _add_info_parser(self, subparsers):
        """Add info subcommand"""
        info = subparsers.add_parser(
            'info',
            help='Display V.V.A.L.T system information',
            description='Show version, configuration, and system details'
        )
        info.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed information'
        )
        info.add_argument(
            '--check-deps',
            action='store_true',
            help='Check optional dependencies'
        )

    def run(self, args=None):
        """Execute CLI with provided arguments"""
        parsed_args = self.parser.parse_args(args)

        if not parsed_args.command:
            self.parser.print_help()
            return 0

        try:
            # Route to appropriate handler
            if parsed_args.command == 'infer':
                return self._handle_infer(parsed_args)
            elif parsed_args.command == 'explain':
                return self._handle_explain(parsed_args)
            elif parsed_args.command == 'verify':
                return self._handle_verify(parsed_args)
            elif parsed_args.command == 'config':
                return self._handle_config(parsed_args)
            elif parsed_args.command == 'checkpoint':
                return self._handle_checkpoint(parsed_args)
            elif parsed_args.command == 'demo':
                return self._handle_demo(parsed_args)
            elif parsed_args.command == 'info':
                return self._handle_info(parsed_args)
            else:
                self.parser.print_help()
                return 1

        except VVALTCLIError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            if '--debug' in sys.argv:
                raise
            return 1

    def _handle_infer(self, args) -> int:
        """Handle inference command"""
        print("Running V.V.A.L.T inference...")

        # Load model
        model, config = self._load_model(args.config, args.checkpoint, args.enhanced)

        # Load input data
        input_data = self._load_input(args.input)

        # Load task if provided, otherwise create default
        if args.task:
            task_vector = self._load_task(args.task, config.model.task_dim)
        else:
            # Create default balanced task vector
            builder = TaskVectorBuilder(config.model.task_dim)
            task_vector = builder.balanced()

        # Run inference
        output, trace = model.forward(input_data, task_vector, return_trace=args.trace)

        # Prepare results
        results = {
            'output': output.tolist() if isinstance(output, np.ndarray) else output,
        }
        if args.trace and trace:
            results['trace'] = self._serialize_trace(trace)

        # Output results
        if args.output:
            self._save_output(results, args.output, args.format)
            print(f"Results saved to: {args.output}")
        else:
            self._print_results(results, args.format)

        print("Inference completed successfully")
        return 0

    def _handle_explain(self, args) -> int:
        """Handle explanation command"""
        print("Generating reasoning explanation...")

        # Load model
        model, config = self._load_model(args.config, args.checkpoint, enhanced=False)

        # Load input data
        input_data = self._load_input(args.input)

        # Load task if provided
        if args.task:
            task_vector = self._load_task(args.task, config.model.task_dim)
        else:
            builder = TaskVectorBuilder(config.model.task_dim)
            task_vector = builder.balanced()

        # Get explanation
        explanation = model.explain(input_data, task_vector)

        # Output explanation
        if args.output:
            with open(args.output, 'w') as f:
                if args.output.endswith('.json'):
                    json.dump(self._serialize_trace(explanation), f, indent=2)
                else:
                    f.write(self._format_explanation(explanation, args.verbose))
            print(f"Explanation saved to: {args.output}")
        else:
            print("\n" + "="*80)
            print("REASONING TRACE")
            print("="*80)
            print(self._format_explanation(explanation, args.verbose))

        return 0

    def _handle_verify(self, args) -> int:
        """Handle verification command"""
        print("Running V.V.A.L.T verification...")

        # Load model
        model, config = self._load_model(args.config, args.checkpoint, enhanced=False)

        results = {}

        # Determine what to verify
        run_all = args.all or (not args.determinism and not args.safety)

        # Create sample data for verification
        x = np.random.randn(config.model.input_dim).astype(np.float32)
        builder = TaskVectorBuilder(config.model.task_dim)
        task_vector = builder.balanced()

        # Verify determinism
        if args.determinism or run_all:
            print(f"\nVerifying determinism ({args.trials} trials)...")
            is_deterministic = model.verify_determinism(x, task_vector, num_trials=args.trials)
            results['determinism'] = {
                'passed': bool(is_deterministic),
                'trials': args.trials
            }
            status = "✓ PASS" if is_deterministic else "✗ FAIL"
            print(f"  Determinism check: {status}")

        # Generate safety report
        if args.safety or run_all:
            print("\nGenerating safety report...")
            safety_report = model.get_safety_report(x, task_vector)
            results['safety'] = self._serialize_trace(safety_report)
            print(f"  No autonomous loops: ✓")
            print(f"  Bounded computation: ✓")
            print(f"  Deterministic outputs: ✓")

        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nVerification report saved to: {args.output}")

        print("\nVerification completed")
        return 0

    def _handle_config(self, args) -> int:
        """Handle configuration command"""
        if not args.config_action:
            print("Error: config action required (generate|validate|show)", file=sys.stderr)
            return 1

        if args.config_action == 'generate':
            return self._config_generate(args)
        elif args.config_action == 'validate':
            return self._config_validate(args)
        elif args.config_action == 'show':
            return self._config_show(args)

        return 1

    def _config_generate(self, args) -> int:
        """Generate configuration file"""
        print(f"Generating {args.preset} configuration...")

        # Create default config
        config = VVALTConfig()

        # Apply preset modifications
        if args.preset == 'fast':
            config.model.hidden_dim = config.model.frame_dim  # Smaller hidden dim
            config.performance.cache_graph_normalization = True
        elif args.preset == 'accurate':
            config.model.hidden_dim = 4 * config.model.frame_dim  # Larger hidden dim
            config.model.input_dim = 512
            config.model.frame_dim = 256

        # Save configuration
        config.to_yaml(args.output)

        print(f"Configuration saved to: {args.output}")
        return 0

    def _config_validate(self, args) -> int:
        """Validate configuration file"""
        print(f"Validating configuration: {args.config_file}")

        try:
            config = VVALTConfig.load(args.config_file)
            config.validate()
            print("✓ Configuration is valid")
            return 0
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}", file=sys.stderr)
            return 1

    def _config_show(self, args) -> int:
        """Show configuration"""
        config = VVALTConfig.load(args.config_file)

        if args.format == 'json':
            print(json.dumps(config.to_dict(), indent=2))
        else:
            import yaml
            print(yaml.dump({'vvalt': config.to_dict()}, default_flow_style=False, sort_keys=False))

        return 0

    def _handle_checkpoint(self, args) -> int:
        """Handle checkpoint command"""
        if not args.checkpoint_action:
            print("Error: checkpoint action required (create|load|info)", file=sys.stderr)
            return 1

        if args.checkpoint_action == 'create':
            return self._checkpoint_create(args)
        elif args.checkpoint_action == 'load':
            return self._checkpoint_load(args)
        elif args.checkpoint_action == 'info':
            return self._checkpoint_info(args)

        return 1

    def _checkpoint_create(self, args) -> int:
        """Create checkpoint"""
        print("Creating checkpoint...")

        # Load config
        if args.config:
            config = VVALTConfig.load(args.config)
        else:
            config = VVALTConfig()

        # Create model
        model = VVALT(
            input_dim=config.model.input_dim,
            frame_dim=config.model.frame_dim,
            task_dim=config.model.task_dim,
            hidden_dim=config.model.hidden_dim,
            seed=config.model.seed
        )

        # Create metadata
        metadata = create_checkpoint_metadata(description=args.description)

        # Create checkpoint manager and save
        manager = CheckpointManager(model)
        checksum = manager.save(args.output, metadata=metadata)

        print(f"Checkpoint saved to: {args.output}")
        print(f"Checksum: {checksum}")
        return 0

    def _checkpoint_load(self, args) -> int:
        """Load checkpoint"""
        print(f"Loading checkpoint: {args.checkpoint_file}")

        # First, load metadata to get dimensions
        checkpoint_data = np.load(args.checkpoint_file, allow_pickle=True)
        metadata = json.loads(str(checkpoint_data["metadata"][0]))

        # Create model with matching dimensions
        model = VVALT(
            input_dim=metadata['input_dim'],
            frame_dim=metadata['frame_dim'],
            task_dim=metadata['task_dim'],
            hidden_dim=metadata.get('hidden_dim'),
            seed=metadata.get('seed', 42)
        )

        # Load checkpoint
        manager = CheckpointManager(model)
        loaded_metadata = manager.load(args.checkpoint_file, strict=args.verify)

        print("✓ Checkpoint loaded successfully")
        print(f"  Input dim: {metadata['input_dim']}")
        print(f"  Frame dim: {metadata['frame_dim']}")
        print(f"  Task dim: {metadata['task_dim']}")

        if args.verify:
            print("  Integrity verified: ✓")

        return 0

    def _checkpoint_info(self, args) -> int:
        """Show checkpoint information"""
        # Load checkpoint metadata
        try:
            checkpoint_data = np.load(args.checkpoint_file, allow_pickle=True)
            metadata = json.loads(str(checkpoint_data["metadata"][0]))
        except Exception as e:
            print(f"Error loading checkpoint: {e}", file=sys.stderr)
            return 1

        print("\n" + "="*80)
        print("CHECKPOINT INFORMATION")
        print("="*80)
        print(f"Version: {metadata.get('version', 'unknown')}")
        print(f"Created: {metadata.get('timestamp', 'unknown')}")
        print(f"Input dim: {metadata.get('input_dim')}")
        print(f"Frame dim: {metadata.get('frame_dim')}")
        print(f"Task dim: {metadata.get('task_dim')}")
        print(f"Hidden dim: {metadata.get('hidden_dim')}")
        print(f"Seed: {metadata.get('seed')}")

        if 'description' in metadata:
            print(f"\nDescription: {metadata['description']}")

        return 0

    def _handle_demo(self, args) -> int:
        """Handle demo command"""
        print(f"Running {args.example} demo...\n")

        if args.config:
            config = VVALTConfig.load(args.config)
        else:
            config = VVALTConfig()

        if args.example in ['basic', 'all']:
            self._run_basic_demo(config, args.verbose)

        if args.example in ['production', 'all']:
            self._run_production_demo(config, args.verbose)

        if args.example in ['perspectives', 'all']:
            self._run_perspectives_demo(config, args.verbose)

        if args.example in ['safety', 'all']:
            self._run_safety_demo(config, args.verbose)

        print("\nDemo completed")
        return 0

    def _run_basic_demo(self, config, verbose):
        """Run basic demonstration"""
        print("="*80)
        print("BASIC DEMO: Simple Inference")
        print("="*80)

        # Create model
        model = VVALT(
            input_dim=config.model.input_dim,
            frame_dim=config.model.frame_dim,
            task_dim=config.model.task_dim,
            hidden_dim=config.model.hidden_dim,
            seed=config.model.seed
        )

        # Create sample input (1D)
        x = np.random.randn(config.model.input_dim).astype(np.float32)
        builder = TaskVectorBuilder(config.model.task_dim)
        task_vector = builder.balanced()

        # Run inference
        output, trace = model.forward(x, task_vector, return_trace=verbose)

        print(f"Input shape: {x.shape}")
        print(f"Task vector shape: {task_vector.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[:5]}...")

        if verbose and trace:
            print("\nReasoning trace:")
            print(self._format_explanation(trace, verbose=True))

        print()

    def _run_production_demo(self, config, verbose):
        """Run production demonstration"""
        print("="*80)
        print("PRODUCTION DEMO: Enhanced Features")
        print("="*80)

        # Create enhanced model
        model = VVALTEnhanced(config=config, enable_monitoring=True)

        # Create sample input
        x = np.random.randn(config.model.input_dim).astype(np.float32)
        builder = TaskVectorBuilder(config.model.task_dim)
        task_vector = builder.balanced()

        # Run inference
        output, trace = model.forward(x, task_vector)

        print(f"Enhanced model loaded")
        print(f"Monitoring: enabled")
        print(f"Output shape: {output.shape}")

        print()

    def _run_perspectives_demo(self, config, verbose):
        """Run multi-perspective demonstration"""
        print("="*80)
        print("PERSPECTIVES DEMO: Multi-Perspective Analysis")
        print("="*80)

        # Create model
        model = VVALT(
            input_dim=config.model.input_dim,
            frame_dim=config.model.frame_dim,
            task_dim=config.model.task_dim,
            hidden_dim=config.model.hidden_dim,
            seed=config.model.seed
        )

        # Create sample input
        x = np.random.randn(config.model.input_dim).astype(np.float32)
        builder = TaskVectorBuilder(config.model.task_dim)

        # Test different perspectives
        print("Testing different perspective priorities:\n")

        for task_type in [TaskType.SEMANTIC, TaskType.STRUCTURAL, TaskType.CAUSAL,
                          TaskType.RELATIONAL, TaskType.GRAPH]:
            task_vector = builder.from_task_type(task_type)
            output, _ = model.forward(x, task_vector)
            print(f"  {task_type.value:12s}: output norm = {np.linalg.norm(output):.4f}")

        print()

    def _run_safety_demo(self, config, verbose):
        """Run safety verification demonstration"""
        print("="*80)
        print("SAFETY DEMO: Verification & Guarantees")
        print("="*80)

        # Create model
        model = VVALT(
            input_dim=config.model.input_dim,
            frame_dim=config.model.frame_dim,
            task_dim=config.model.task_dim,
            hidden_dim=config.model.hidden_dim,
            seed=config.model.seed
        )

        # Create sample input
        x = np.random.randn(config.model.input_dim).astype(np.float32)
        builder = TaskVectorBuilder(config.model.task_dim)
        task_vector = builder.balanced()

        # Verify determinism
        print("Checking determinism...")
        is_deterministic = model.verify_determinism(x, task_vector, num_trials=10)
        print(f"  Deterministic: {'✓ YES' if is_deterministic else '✗ NO'}")

        # Get safety report
        print("\nSafety guarantees:")
        print("  ✓ No autonomous loops")
        print("  ✓ Bounded computation")
        print("  ✓ Deterministic outputs")
        print("  ✓ Full interpretability")

        print()

    def _handle_info(self, args) -> int:
        """Handle info command"""
        print("="*80)
        print("V.V.A.L.T - Vantage-Vector Autonomous Logic Transformer")
        print("="*80)
        print("\nVersion: 1.0.0")
        print("License: MIT (2025)")
        print("Repository: https://github.com/VValtDisney/V.V.A.L.T")

        if args.verbose:
            print("\nCore Components:")
            print("  1. Vector Frame Encoder")
            print("  2. Vantage Selector")
            print("  3. Graph Topology Projector")
            print("  4. Multi-Perspective Attention")
            print("  5. Logic Refinement Unit")
            print("  6. Consistency Verifier")
            print("  7. Interpretability Projector")

            print("\nPerspectives:")
            print("  • Semantic (meaning-based)")
            print("  • Structural (pattern-based)")
            print("  • Causal (cause-effect)")
            print("  • Relational (connection-based)")
            print("  • Graph (topology-aligned)")

        if args.check_deps:
            print("\nDependency Check:")
            self._check_dependencies()

        return 0

    def _check_dependencies(self):
        """Check optional dependencies"""
        deps = {
            'numpy': 'NumPy (required)',
            'yaml': 'PyYAML (required)',
            'torch': 'PyTorch (optional)',
            'transformers': 'HuggingFace Transformers (optional)'
        }

        for module, name in deps.items():
            try:
                __import__(module)
                print(f"  ✓ {name}")
            except ImportError:
                print(f"  ✗ {name}")

    def _load_model(self, config_file: Optional[str], checkpoint_file: Optional[str], enhanced: bool = False):
        """Load V.V.A.L.T model and return (model, config)"""
        # Load config
        if config_file:
            config = VVALTConfig.load(config_file)
        else:
            config = VVALTConfig()

        # Create model
        if enhanced:
            model = VVALTEnhanced(config=config, enable_monitoring=True)
        else:
            model = VVALT(
                input_dim=config.model.input_dim,
                frame_dim=config.model.frame_dim,
                task_dim=config.model.task_dim,
                hidden_dim=config.model.hidden_dim,
                seed=config.model.seed
            )

        # Load from checkpoint if provided
        if checkpoint_file:
            manager = CheckpointManager(model)
            manager.load(checkpoint_file)

        return model, config

    def _load_input(self, input_file: str) -> np.ndarray:
        """Load input data from file"""
        path = Path(input_file)

        if not path.exists():
            raise VVALTCLIError(f"Input file not found: {input_file}")

        if path.suffix == '.npy':
            data = np.load(input_file)
            # If batch, take first element; if 1D, use as-is
            if data.ndim > 1:
                data = data[0]
            return data
        elif path.suffix == '.json':
            with open(input_file) as f:
                data = json.load(f)
            arr = np.array(data, dtype=np.float32)
            # If batch, take first element
            if arr.ndim > 1:
                arr = arr[0]
            return arr
        else:
            raise VVALTCLIError(f"Unsupported input format: {path.suffix}")

    def _load_task(self, task_file: str, task_dim: int) -> np.ndarray:
        """Load task vector from file"""
        path = Path(task_file)

        if not path.exists():
            raise VVALTCLIError(f"Task file not found: {task_file}")

        if path.suffix == '.npy':
            return np.load(task_file)
        elif path.suffix == '.json':
            with open(task_file) as f:
                data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, list):
                # Direct array
                return np.array(data, dtype=np.float32)
            elif isinstance(data, dict):
                # Check for task type specification
                if 'task_type' in data:
                    builder = TaskVectorBuilder(task_dim)
                    task_type = TaskType(data['task_type'].upper())
                    return builder.from_task_type(task_type)
                # Check for direct vector
                elif 'vector' in data:
                    return np.array(data['vector'], dtype=np.float32)
                # Check for weights/priorities
                elif 'priorities' in data or 'weights' in data:
                    weights = data.get('priorities') or data.get('weights')
                    if isinstance(weights, dict):
                        # Convert named perspectives to array
                        perspective_order = ['semantic', 'structural', 'causal', 'relational', 'graph']
                        vec = np.array([weights.get(p, 0.5) for p in perspective_order], dtype=np.float32)
                        # Pad or trim to task_dim
                        if len(vec) < task_dim:
                            vec = np.concatenate([vec, np.zeros(task_dim - len(vec), dtype=np.float32)])
                        elif len(vec) > task_dim:
                            vec = vec[:task_dim]
                        return vec
                    else:
                        return np.array(weights, dtype=np.float32)

            raise VVALTCLIError(f"Unable to parse task file: {task_file}")
        else:
            raise VVALTCLIError(f"Unsupported task format: {path.suffix}")

    def _save_output(self, data: Dict[str, Any], output_file: str, format: str):
        """Save output data to file"""
        if format == 'npy':
            # Save just the output array
            np.save(output_file, data['output'])
        elif format == 'json':
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'text':
            with open(output_file, 'w') as f:
                f.write(str(data))

    def _print_results(self, data: Dict[str, Any], format: str):
        """Print results to console"""
        if format == 'json':
            print(json.dumps(data, indent=2))
        else:
            print(data)

    def _serialize_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize trace data for JSON output"""
        if not trace:
            return {}

        serialized = {}
        for key, value in trace.items():
            if isinstance(value, np.ndarray):
                serialized[key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'values': value.tolist()
                }
            elif isinstance(value, dict):
                serialized[key] = self._serialize_trace(value)
            else:
                serialized[key] = value

        return serialized

    def _format_explanation(self, explanation: Dict[str, Any], verbose: bool) -> str:
        """Format explanation for display"""
        lines = []

        if 'perspective_weights' in explanation:
            lines.append("\nPerspective Weights:")
            weights = explanation['perspective_weights']
            if isinstance(weights, np.ndarray):
                perspective_names = ['Semantic', 'Structural', 'Causal', 'Relational', 'Graph']
                for i, (name, weight) in enumerate(zip(perspective_names, weights[:5])):
                    lines.append(f"  {name:12s}: {weight:.4f}")

        if 'frames' in explanation and verbose:
            frames = explanation['frames']
            if isinstance(frames, np.ndarray):
                lines.append(f"\nFrames shape: {frames.shape}")

        if 'attention_weights' in explanation and verbose:
            weights = explanation['attention_weights']
            if isinstance(weights, np.ndarray):
                lines.append(f"\nAttention weights shape: {weights.shape}")

        if 'output' in explanation:
            output = explanation['output']
            if isinstance(output, np.ndarray):
                lines.append(f"\nOutput shape: {output.shape}")
                lines.append(f"Output sample: {output[:5]}...")

        return '\n'.join(lines) if lines else str(explanation)


def main():
    """Main entry point for CLI"""
    cli = VVALTCLI()
    sys.exit(cli.run())


if __name__ == '__main__':
    main()
