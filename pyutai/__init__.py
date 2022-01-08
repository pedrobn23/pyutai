import sys
from typing import Dict, Iterable, List, Tuple

# Union types are only allowed from python 3.10 onwards [1].
# Typing analisys for access will only be done for List[int]
# for eariler versions.
#
# [1] https://www.python.org/dev/peps/pep-0604/
if sys.version_info.major >= 3 and sys.version_info.major >= 10:
    IndexType = List[int] | Tuple[int]
else:
    IndexType = List[int]
