from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


# Use slots to save memory if Python version supports it.
_dataclass_kwargs = {}
if sys.version_info >= (3, 10):
    _dataclass_kwargs["slots"] = True


@dataclass(**_dataclass_kwargs)
class Transcript:
    """Represents an audio transcript with optional metadata.

    Attributes:
        config: Dictionary of key-value pairs parsed from the transcript header.
        paragraphs: List of paragraphs, where each paragraph is a list of lines.
    """

    config: Optional[Dict[str, str]] = None
    paragraphs: Optional[List[List[str]]] = None

    @classmethod
    def from_lines(
        cls,
        lines: Iterable[str],
        parse_config: bool = True,
    ) -> Transcript:
        """Create a Transcript instance from an iterable of lines.

        Parses optional config key-value pairs (if enabled) followed by paragraphs
        separated by blank lines.

        Args:
            lines: An iterable of strings representing lines of text.
            parse_config: Whether to parse config lines at the top of the input.

        Returns:
            A Transcript instance containing the parsed config and paragraphs.
        """
        config: Optional[Dict[str, str]] = None
        paragraphs: Optional[List[List[str]]] = None
        current_paragraph: List[str] = []
        in_text_section = not parse_config  # Start instantly if config disabled.

        for line in lines:
            line = line.rstrip()

            if not in_text_section:
                if line == "":
                    continue  # Skip leading empty lines.
                if line.startswith("- "):
                    # Config keys are always lowercase, must always begin with
                    # a normal letter first, and then either letters or numbers.
                    m = re.match(r"^- ([a-z][a-z0-9]*):\s*(.+)$", line)
                    if m:
                        if config is None:
                            config = {}
                        config[m.group(1)] = m.group(2).rstrip()
                        continue
                # First non-config line starts the text section.
                in_text_section = True

            # Handle paragraph accumulation.
            if line == "":
                if current_paragraph:
                    if paragraphs is None:
                        paragraphs = []
                    paragraphs.append(current_paragraph)
                    current_paragraph = []
            else:
                current_paragraph.append(line)

        # Append final paragraph (if it wasn't already handled above).
        if current_paragraph:
            if paragraphs is None:
                paragraphs = []
            paragraphs.append(current_paragraph)

        return cls(config=config, paragraphs=paragraphs)

    @classmethod
    def from_file(
        cls,
        filename: Union[str, Path],
        parse_config: bool = True,
    ) -> Transcript:
        """Create a Transcript instance by reading from a text file.

        Args:
            filename: Path to the input text file.
            parse_config: Whether to parse config lines at the top of the file.

        Returns:
            A Transcript instance containing the parsed config and paragraphs.
        """
        with open(filename, "rt", encoding="utf-8") as f:
            return cls.from_lines(f, parse_config)

    @classmethod
    def from_text(
        cls,
        raw_text: str,
        parse_config: bool = False,
    ) -> Transcript:
        """Create a Transcript instance from a raw text string.

        Args:
            raw_text: The full input text as a single string.
            parse_config: Whether to parse config lines at the top of the input.

        Returns:
            A Transcript instance containing the parsed config and paragraphs.
        """
        return cls.from_lines(raw_text.splitlines(), parse_config)

    def as_text(self) -> str:
        """Serialize the Transcript instance into a formatted string.

        Outputs config key-value pairs (if any) followed by paragraphs,
        with blank lines separating the paragraphs.

        Returns:
            A string representation of the transcript.
        """
        lines: List[str] = []

        if self.config:
            for key, value in self.config.items():
                lines.append(f"- {key}: {value}")

        if self.paragraphs:
            if self.config:
                lines.append("")  # Blank line between config and paragraphs.

            last_i = len(self.paragraphs) - 1
            for i, paragraph in enumerate(self.paragraphs):
                lines.extend(paragraph)
                if i < last_i:
                    lines.append("")  # Blank line between paragraphs.

        return "\n".join(lines)
