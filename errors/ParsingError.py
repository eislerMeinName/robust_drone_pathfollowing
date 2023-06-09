from typing import List


class ParsingError(Exception):
    """Exception raised when the patsed Arguments do not match."""

    def __init__(self, args: List, value: List, message: str):
        """Initialization of ParseMatchError."""

        msg: str = ''
        for i, arg in enumerate(args):
            msg += ' ' + str(arg) + '(' + str(value[i]) + ')'
            msg += ','

        msg = msg[0:len(msg) - 1]
        msg += '. ' + message
        super().__init__(msg)