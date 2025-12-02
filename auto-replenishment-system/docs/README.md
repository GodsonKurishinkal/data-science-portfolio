# Documentation

This directory contains documentation for the Universal Replenishment Engine.

## Contents

- [Architecture Guide](architecture.md) - System architecture and design
- [API Reference](api-reference.md) - Module and class documentation
- [Configuration Guide](configuration.md) - YAML configuration reference
- [Scenarios Guide](scenarios.md) - Supported retail scenarios

## Quick Links

- [Quick Start](../quick-start.md) - Get started in 5 minutes
- [Demo Script](../demo.py) - Interactive demonstration
- [Configuration](../config/config.yaml) - Main configuration file

## Key Concepts

### Periodic Review (s,S) Policy

The primary inventory policy where:
- `s` (Reorder Point) = DDR × LT + Safety Stock
- `S` (Order-Up-To Level) = DDR × (LT + RP) + Safety Stock
- Order Quantity = S - Inventory Position (when IP ≤ s)

### ABC/XYZ Classification

- **ABC**: Value-based (A=80%, B=15%, C=5% of revenue)
- **XYZ**: Variability-based (CV < 0.5 = X, 0.5-1.0 = Y, > 1.0 = Z)
- Combined 9-cell matrix determines service levels

### Safety Stock Formula

**Standard**: SS = Z × σ_demand × √LT

**Dynamic**: SS = Z × √(LT × σ²_demand + DDR² × σ²_LT)

### Service Level Matrix

| | X (Stable) | Y (Variable) | Z (Erratic) |
|---|---|---|---|
| **A (High Value)** | 99% | 97% | 95% |
| **B (Medium Value)** | 97% | 95% | 92% |
| **C (Low Value)** | 95% | 92% | 90% |

## Business Impact

- 20-30% inventory reduction
- 95-99% service level achievement
- 15-25% cost savings
- 50%+ reduction in stockouts
