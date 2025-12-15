# Exceptions that the user can fix and restart. We will not mark the assignment as failed.
class RecoverableException(Exception):
    pass

class InvalidModelParametersException(Exception):
    pass
