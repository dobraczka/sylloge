import tempfile
from nox_poetry import Session, session

def install_poetry_groups(session, *groups: str) -> None:
    """Install dependencies from poetry groups.

    Using this as s workaround until my PR is merged in:
    https://github.com/cjolowicz/nox-poetry/pull/1080
    """
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            *[f"--only={group}" for group in groups],
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install("-r", requirements.name)


@session()
def tests(session: Session) -> None:
    args = session.posargs or ["--cov", "--cov-report=xml"]
    session.install(".")
    install_poetry_groups(session, "dev")
    session.run("pytest", *args)


locations = ["sylloge", "tests", "noxfile.py"]


@session(tags=["not test", "style"])  # type: ignore
def lint(session: Session) -> None:
    args = session.posargs or locations
    session.install("black", "isort")
    session.run("black", *args)
    session.run("isort", *args)


@session(tags=["not test", "style"])  # type: ignore
def style_checking(session: Session) -> None:
    args = session.posargs or locations
    session.install(
        "pyproject-flake8",
        "flake8-eradicate",
        "flake8-isort",
        "flake8-debugger",
        "flake8-comprehensions",
        "flake8-print",
        "flake8-black",
        "flake8-black",
        "darglint",
        "pydocstyle",
    )
    session.run("pflake8", "--docstring-style", "sphinx", *args)


@session(tags=["not test", "style"])  # type: ignore
def type_checking(session: Session) -> None:
    args = session.posargs or locations
    session.run_always("poetry", "install", external=True)
    session.run(
        "mypy",
        "--install-types",
        "--ignore-missing-imports",
        "--non-interactive",
        *args,
    )


@session(tags=["not test"])  # type: ignore
def build_docs(session: Session) -> None:
    session.install(".[docs]")
    session.install("sphinx")
    session.install("insegel")
    session.install("sphinx-automodapi")
    session.install("sphinx-autodoc-typehints")
    session.cd("docs")
    session.run("make", "clean", external=True)
    session.run("make", "html", external=True)
