import json
import subprocess
from pathlib import Path

from pydantic import BaseModel


class RawIssue(BaseModel):
    """Represents a raw issue loaded from a local file or GitHub.

    This is the initial unprocessed form of an issue before normalization.
    """
    title: str
    body: str
    source: str


def load_local_issue(path: str) -> RawIssue:
    """Load an issue from a local markdown file.

    Args:
        path: Path to the local issue file.

    Returns:
        RawIssue with title extracted from the first heading or filename.

    Raises:
        FileNotFoundError: If the issue file does not exist.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Issue file not found: {path}")

    body = file_path.read_text(encoding="utf-8")

    # Default to using filename as title
    title = file_path.stem

    # Use first line starting with "# " as title if available
    for line in body.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    return RawIssue(
        title=title,
        body=body,
        source=str(file_path),
    )


def load_github_issue(url: str) -> RawIssue:
    """Load an issue from GitHub using the gh CLI.

    Args:
        url: GitHub issue URL.

    Returns:
        RawIssue with title, body, and comments appended.

    Raises:
        RuntimeError: If the gh CLI command fails.
    """
    result = subprocess.run(
        [
            "gh",
            "issue",
            "view",
            url,
            "--json",
            "title,body,comments",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to load GitHub issue:\n{result.stderr}"
        )

    data = json.loads(result.stdout)

    body = data.get("body", "")

    # Comments may contain additional requirements
    comments = data.get("comments", [])

    if comments:
        body += "\n\n## Comments\n"

        for comment in comments:
            author = comment.get("author", {}).get("login", "unknown")
            comment_body = comment.get("body", "")

            body += f"\n### {author}\n{comment_body}\n"

    return RawIssue(
        title=data["title"],
        body=body,
        source=url,
    )


def load_issue(source: str) -> RawIssue:
    """Load an issue from either a local file or GitHub URL.

    Args:
        source: Local file path or GitHub issue URL.

    Returns:
        RawIssue loaded from the appropriate source.
    """
    if source.startswith(
        ("https://github.com/", "http://github.com/")
    ):
        return load_github_issue(source)

    return load_local_issue(source)
