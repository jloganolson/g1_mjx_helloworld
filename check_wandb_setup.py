import sys
import wandb
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file
def check_wandb():
    """Checks if wandb is installed and the user is logged in."""
    print("Checking wandb installation...")
    try:
        print(f"Successfully imported wandb version: {wandb.__version__}")
    except ImportError:
        print("Error: wandb is not installed in this environment.", file=sys.stderr)
        print("Please install it using: pip install wandb", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during import: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nChecking wandb login status...")
    # Use wandb.login(anonymous="never") to check status without prompting
    login_status = wandb.login(anonymous="never", timeout=10) # Added timeout

    if login_status:
        print("Successfully logged in to wandb.")
        try:
            # Verify login with a simple API call if possible
            viewer = wandb.api.viewer()
            if viewer:
                 print(f"Logged in as user: {viewer.get('username', 'N/A')}, Default entity: {viewer.get('entity', 'N/A')}")
                 print("W&B API connection verified.")
            else:
                 print("Could not retrieve user info via API, but login status reported success.")
        except Exception as e:
            print(f"Warning: Could not verify W&B API connection. Login might be stale or network issues present: {e}", file=sys.stderr)

    else:
        print("Error: Not logged in to wandb or login check failed.", file=sys.stderr)
        print("Please log in using the command: wandb login", file=sys.stderr)
        sys.exit(1)

    print("\nWandb check complete.")

if __name__ == "__main__":
    check_wandb()
