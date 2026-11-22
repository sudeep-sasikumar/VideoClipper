"""Errors with a message written for the user, not a stack trace."""


class ClipperError(Exception):
    """Base error. str(err) is shown to the user as-is."""

    title = "Error"

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n{self.hint}" if self.hint else self.message


class BadLinkError(ClipperError):
    title = "Bad link"


class ToolMissingError(ClipperError):
    title = "Missing tool"


class ClaudeNotFoundError(ToolMissingError):
    title = "Claude CLI not found"


class ClaudeNotLoggedInError(ClipperError):
    title = "Claude CLI not logged in"


class ClaudeOutputError(ClipperError):
    title = "Claude returned bad output"


class DiskSpaceError(ClipperError):
    title = "Not enough disk space"


class Cancelled(ClipperError):
    title = "Cancelled"

    def __init__(self, message: str = "Cancelled by user."):
        super().__init__(message)
