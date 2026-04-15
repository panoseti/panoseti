from pathlib import Path

# Try importing the library; handle failure gracefully for legacy systems
try:
    from panoseti_grpc.telemetry.config import TelemetryConfig
except ImportError:
    TelemetryConfig = None


class TelemetryConfigManager:
    """
    Manages the dynamic configuration for the Telemetry Service.
    Handles hot-reloading of the toml file and matching Redis keys to device modes.
    """

    def __init__(self, config_dir=None):
        # 1. Determine Config Path
        # Default: look in the same directory as this util file
        if config_dir:
            self.config_path = Path(config_dir) / "telemetry_config.toml"
        else:
            self.config_path = Path(__file__).parent / "telemetry_config.toml"

        self.last_mtime = 0
        self.config = None
        self.active_prefixes = {}  # Cache for fast lookups

        if TelemetryConfig:
            self.reload()
        else:
            print("[TelemeteryService] Library not found. Dynamic features disabled.")

    def reload(self):
        """Checks disk for changes and reloads if necessary."""
        if not TelemetryConfig: return

        try:
            if not self.config_path.exists():
                return

            mtime = self.config_path.stat().st_mtime
            if mtime != self.last_mtime:
                print(f"[TelemeteryService] Reloading config from: {self.config_path}")
                self.config = TelemetryConfig.load(str(self.config_path))
                self.last_mtime = mtime

                # Rebuild prefix cache for O(1) lookup in the tight loop
                self.active_prefixes = {}
                for device_type, dev_cfg in self.config.devices.items():
                    self.active_prefixes[dev_cfg.redis_prefix] = (device_type, dev_cfg.mode)

                print(f"[TelemeteryService] Active Devices: {len(self.active_prefixes)}")
        except Exception as e:
            print(f"[TelemeteryService] Config Load Error: {e}")

    def match_key(self, redis_key):
        """
        Matches a Redis Key to a (datatype, mode) tuple.
        Returns (None, None) if no match.
        """
        if not self.config: return None, None

        # Fast prefix check
        # Note: This simple implementation assumes prefixes don't overlap in ambiguous ways
        for prefix, (dtype, mode) in self.active_prefixes.items():
            if redis_key.startswith(prefix):
                return dtype, mode
        return None, None