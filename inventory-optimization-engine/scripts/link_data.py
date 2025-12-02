#!/usr/bin/env python3
"""Script to create symlink to M5 data from project-001."""

import os
from pathlib import Path


def main():
    """Create symlink to M5 data."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'

    # Source data from project-001
    source_data = project_root.parent / 'project-001-demand-forecasting-system' / 'data' / 'raw'

    # Target symlink
    target_link = data_dir / 'raw'

    if target_link.exists():
        if target_link.is_symlink():
            print(f"✅ Symlink already exists: {target_link} -> {os.readlink(target_link)}")
        else:
            print(f"⚠️  {target_link} exists but is not a symlink")
        return

    if not source_data.exists():
        print(f"❌ Source data not found: {source_data}")
        print("   Please ensure project-001 data is available")
        return

    # Create symlink
    try:
        target_link.symlink_to(source_data, target_is_directory=True)
        print(f"✅ Created symlink: {target_link} -> {source_data}")
        print("   M5 data is now accessible in this project")
    except Exception as e:
        print(f"❌ Failed to create symlink: {e}")
        print("   You may need to copy the data manually")


if __name__ == '__main__':
    main()
