"""Vahtian namespace package.

This package may be installed alongside sibling Vahtian tools.  Keep the
legacy namespace extension so source checkouts remain importable even when
another installed ``vahtian`` distribution has already claimed the package.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
