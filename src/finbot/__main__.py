"""Enable ``python -m finbot`` to invoke the command line interface."""

import sys

from finbot.cli import main

if __name__ == "__main__":
    sys.exit(main())
