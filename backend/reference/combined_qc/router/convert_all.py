"""Router conversion entry point; the public CLI is python -m combined_qc.router."""
from .conversion import convert_all

__all__ = ['convert_all']


if __name__ == '__main__':
    import sys
    from .__main__ import main
    main(['convert', *sys.argv[1:]])
