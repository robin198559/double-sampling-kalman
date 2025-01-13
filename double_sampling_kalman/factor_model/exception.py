class GradientDescentOutOfBoundError(Exception):
    def __init__(self, message):
        super().__init__(message)


class GradientDescentExhaustedError(Exception):
    def __init__(self, message):
        super().__init__(message)
