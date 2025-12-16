"""Constellation configuration utilities."""

import os
import sys
from typing import Optional, Tuple

def get_constellation_ids(
    cli_org_id: Optional[str] = None,
    cli_entity_id: Optional[str] = None
) -> Tuple[str, str]:
    """Get organization_id and entity_id from CLI args, environment, or user input.

    Priority order:
    1. CLI arguments (--org-id, --entity-id)
    2. Environment variables (CONSTELLATION_ORG_ID, CONSTELLATION_ENTITY_ID)
    3. Interactive user input
    """
    print("\n=== Constellation Configuration ===")
    print("Initializing Constellation Overwatch Edge Awareness connection...")
    print()

    # Try to get organization_id: CLI > env > input
    org_id = (cli_org_id or os.environ.get('CONSTELLATION_ORG_ID') or '').strip()
    if not org_id:
        print("Organization ID not found in environment (CONSTELLATION_ORG_ID)")
        print("Please obtain your Organization ID from:")
        print("  - Constellation Overwatch Edge Awareness Kit UI")
        print("  - Your Database Administrator")
        print()
        org_id = input("Enter Organization ID: ").strip()
        if not org_id:
            print("Error: Organization ID is required")
            sys.exit(1)
    else:
        source = "CLI flag" if cli_org_id else "environment"
        print(f"Organization ID loaded from {source}: {org_id}")

    # Try to get entity_id: CLI > env > input
    ent_id = (cli_entity_id or os.environ.get('CONSTELLATION_ENTITY_ID') or '').strip()
    if not ent_id:
        print("Entity ID not found in environment (CONSTELLATION_ENTITY_ID)")
        print("Please obtain your Entity ID from:")
        print("  - Constellation Overwatch Edge Awareness Kit UI")
        print("  - Your Database Administrator")
        print()
        ent_id = input("Enter Entity ID: ").strip()
        if not ent_id:
            print("Error: Entity ID is required")
            sys.exit(1)
    else:
        source = "CLI flag" if cli_entity_id else "environment"
        print(f"Entity ID loaded from {source}: {ent_id}")

    print("===================================\n")
    return org_id, ent_id