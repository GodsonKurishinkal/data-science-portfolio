"""
Configuration Loader for Supply Chain Planning System.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModuleConfig:
    """Base configuration for a planning module."""
    enabled: bool = True
    path: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemandConfig(ModuleConfig):
    """Configuration for demand forecasting module."""
    horizon_days: int = 90
    models: list = field(default_factory=lambda: ['lightgbm', 'prophet'])
    features: list = field(default_factory=lambda: ['lag', 'rolling', 'calendar'])


@dataclass
class InventoryConfig(ModuleConfig):
    """Configuration for inventory optimization module."""
    service_level: float = 0.95
    abc_thresholds: Dict[str, float] = field(default_factory=lambda: {'A': 0.8, 'B': 0.95})
    review_period_days: int = 7


@dataclass
class PricingConfig(ModuleConfig):
    """Configuration for dynamic pricing module."""
    elasticity_method: str = 'log-log'
    optimization_objective: str = 'revenue'
    min_margin: float = 0.10


@dataclass
class NetworkConfig(ModuleConfig):
    """Configuration for network optimization module."""
    solver: str = 'ortools'
    max_distance_km: float = 500.0
    capacity_utilization_target: float = 0.85


@dataclass
class SensingConfig(ModuleConfig):
    """Configuration for real-time sensing module."""
    refresh_interval_seconds: int = 300
    anomaly_threshold: float = 2.0
    alert_channels: list = field(default_factory=lambda: ['dashboard', 'email'])


@dataclass
class ReplenishmentConfig(ModuleConfig):
    """Configuration for auto-replenishment module."""
    default_policy: str = 'periodic_review'
    scenarios: list = field(default_factory=lambda: ['dc_to_store', 'supplier_to_dc'])
    auto_approve_threshold: float = 10000.0


@dataclass
class PlanningConfig:
    """
    Master configuration for the Supply Chain Planning System.
    
    Loads and manages configuration for all integrated modules.
    """
    
    demand: DemandConfig = field(default_factory=DemandConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    replenishment: ReplenishmentConfig = field(default_factory=ReplenishmentConfig)
    
    # Global settings
    data_path: str = "data"
    output_path: str = "output"
    log_level: str = "INFO"
    parallel_execution: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PlanningConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            PlanningConfig instance
        """
        path = Path(config_path)
        
        if not path.exists():
            logger.warning("Config file %s not found, using defaults", config_path)
            return cls()
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'PlanningConfig':
        """Create config from dictionary."""
        config = cls()
        
        # Load module configs
        if 'demand' in data:
            config.demand = DemandConfig(**data['demand'])
        if 'inventory' in data:
            config.inventory = InventoryConfig(**data['inventory'])
        if 'pricing' in data:
            config.pricing = PricingConfig(**data['pricing'])
        if 'network' in data:
            config.network = NetworkConfig(**data['network'])
        if 'sensing' in data:
            config.sensing = SensingConfig(**data['sensing'])
        if 'replenishment' in data:
            config.replenishment = ReplenishmentConfig(**data['replenishment'])
        
        # Load global settings
        if 'data_path' in data:
            config.data_path = data['data_path']
        if 'output_path' in data:
            config.output_path = data['output_path']
        if 'log_level' in data:
            config.log_level = data['log_level']
        if 'parallel_execution' in data:
            config.parallel_execution = data['parallel_execution']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'demand': self._module_to_dict(self.demand),
            'inventory': self._module_to_dict(self.inventory),
            'pricing': self._module_to_dict(self.pricing),
            'network': self._module_to_dict(self.network),
            'sensing': self._module_to_dict(self.sensing),
            'replenishment': self._module_to_dict(self.replenishment),
            'data_path': self.data_path,
            'output_path': self.output_path,
            'log_level': self.log_level,
            'parallel_execution': self.parallel_execution,
        }
    
    def _module_to_dict(self, module: ModuleConfig) -> Dict[str, Any]:
        """Convert module config to dictionary."""
        return {
            'enabled': module.enabled,
            'path': module.path,
            'config': module.config,
        }
    
    def save_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        logger.info("Configuration saved to %s", config_path)


class ConfigLoader:
    """Utility class for loading configurations."""
    
    @staticmethod
    def load(config_path: str) -> PlanningConfig:
        """Load configuration from file."""
        return PlanningConfig.from_yaml(config_path)
    
    @staticmethod
    def load_module_config(config_path: str, module: str) -> Optional[ModuleConfig]:
        """Load configuration for a specific module."""
        config = PlanningConfig.from_yaml(config_path)
        return getattr(config, module, None)
